from __future__ import annotations

from dataclasses import dataclass

from fabweave.core.decision import Decision, deny
from fabweave.holds.release import HoldState, release_decision
from fabweave.spc.control import western_electric_rules


@dataclass(frozen=True, slots=True)
class ExcursionWindow:
    """Suspect interval from the last confirmed-good point to detection.

    A ``last_good_at`` of ``None`` means no observation in the series was
    confirmed good, so every exposure up to ``detected_at`` is suspect.
    """

    last_good_at: int | None
    detected_at: int


def suspect_window(observations: list[tuple[int, float]], center: float, sigma: float) -> ExcursionWindow | None:
    """Locate the last confirmed-good point before the SPC signal.

    Returns ``None`` when the full series trips no western-electric rule.
    Identical readings at the same event time are deduplicated; conflicting
    readings at the same event time are rejected.
    """

    by_time: dict[int, float] = {}
    for measured_at, value in observations:
        previous = by_time.get(measured_at)
        if previous is not None and previous != value:
            raise ValueError("conflicting readings at the same event time")
        by_time[measured_at] = value
    ordered = sorted(by_time.items())
    if not ordered:
        return None
    values = [value for _, value in ordered]
    if not western_electric_rules(values, center, sigma):
        return None
    last_good_at: int | None = None
    for length in range(len(values) - 1, 0, -1):
        if not western_electric_rules(values[:length], center, sigma):
            last_good_at = ordered[length - 1][0]
            break
    return ExcursionWindow(last_good_at, ordered[-1][0])


@dataclass(frozen=True, slots=True)
class StepExposure:
    """One lot observed at one route step at one event time."""

    lot_id: str
    step: str
    at: int


def exposed_lots(exposures: list[StepExposure], step: str, window: ExcursionWindow) -> tuple[str, ...]:
    """Lots that passed the suspect step after the last confirmed-good point."""

    return tuple(
        sorted(
            {
                exposure.lot_id
                for exposure in exposures
                if exposure.step == step
                and (window.last_good_at is None or exposure.at > window.last_good_at)
                and exposure.at <= window.detected_at
            }
        )
    )


class ExcursionCase:
    """Monotonic containment state for one yield excursion.

    The scope and required dispositions are fixed when the case opens.
    Duplicate or late events are absorbed without changing state: the scope
    never grows or shrinks, completed dispositions are never removed, and
    retest recovery never reverts.
    """

    RETEST_BLOCKER = "RETEST_RECOVERY"

    def __init__(
        self,
        case_id: str,
        scope: tuple[str, ...],
        required_dispositions: set[str],
        window: ExcursionWindow,
        opened_at: int,
    ) -> None:
        self._case_id = case_id
        self._scope = tuple(sorted(scope))
        self._required = frozenset(required_dispositions)
        self._window = window
        self._opened_at = opened_at
        self._as_of = opened_at
        self._completed: set[str] = set()
        self._retest_recovered = False
        self._version = 1
        self._events: dict[str, tuple[int, str, object]] = {}

    @property
    def case_id(self) -> str:
        return self._case_id

    @property
    def scope(self) -> tuple[str, ...]:
        return self._scope

    @property
    def window(self) -> ExcursionWindow:
        return self._window

    @property
    def required_dispositions(self) -> frozenset[str]:
        return self._required

    @property
    def completed_dispositions(self) -> frozenset[str]:
        return frozenset(self._completed)

    @property
    def pending_dispositions(self) -> tuple[str, ...]:
        return tuple(sorted(self._required - self._completed))

    @property
    def retest_recovered(self) -> bool:
        return self._retest_recovered

    @property
    def version(self) -> int:
        return self._version

    @property
    def opened_at(self) -> int:
        return self._opened_at

    @property
    def as_of(self) -> int:
        return self._as_of

    @property
    def releasable(self) -> bool:
        return not self.pending_dispositions and self._retest_recovered

    def apply(self, event_id: str, occurred_at: int, kind: str, detail: object = "") -> bool:
        """Apply one event; return True only when the event advanced the case.

        Replaying an identical event is an idempotent no-op, reusing an event
        id with different content is rejected, and an event older than the
        case watermark is absorbed without expanding, shrinking or
        regressing state.
        """

        known = self._events.get(event_id)
        record = (occurred_at, kind, detail)
        if known is not None:
            if known != record:
                raise ValueError("event id was reused with different content")
            return False
        if kind == "disposition_completed":
            if detail not in self._required:
                raise ValueError(f"unknown disposition: {detail}")
        elif kind != "retest_recovered":
            raise ValueError(f"unknown excursion event: {kind}")
        self._events[event_id] = record
        if occurred_at < self._as_of:
            return False
        if kind == "disposition_completed":
            self._completed.add(detail)
        else:
            self._retest_recovered = True
        self._as_of = occurred_at
        self._version += 1
        return True

    def hold_state(self, lot_id: str) -> HoldState:
        """Derive the effective hold for one scoped lot at the current version."""

        if lot_id not in self._scope:
            raise ValueError("lot outside containment scope")
        blockers = frozenset() if self._retest_recovered else frozenset({self.RETEST_BLOCKER})
        return HoldState(
            f"{self._case_id}:{lot_id}",
            self._version,
            self._required,
            frozenset(self._completed),
            blockers,
        )

    def release_decision(self, lot_id: str, expected_version: int, at: int) -> Decision:
        """Release is allowed only once every disposition is complete and the
        retest has recovered; lots outside the scope stay fail-closed."""

        if lot_id not in self._scope:
            return deny("lot_outside_containment_scope", case_id=self._case_id, lot_id=lot_id)
        return release_decision(self.hold_state(lot_id), expected_version, at)


class ExcursionRegister:
    """One case per excursion; reopening the same case id replays idempotently."""

    def __init__(self) -> None:
        self._cases: dict[str, ExcursionCase] = {}

    def open(
        self,
        case_id: str,
        scope: tuple[str, ...],
        required_dispositions: set[str],
        window: ExcursionWindow,
        opened_at: int,
    ) -> tuple[ExcursionCase, bool]:
        existing = self._cases.get(case_id)
        if existing is not None:
            same = (
                existing.scope == tuple(sorted(scope))
                and existing.required_dispositions == frozenset(required_dispositions)
                and existing.window == window
                and existing.opened_at == opened_at
            )
            if not same:
                raise ValueError("case id was reused with different content")
            return existing, False
        case = ExcursionCase(case_id, scope, required_dispositions, window, opened_at)
        self._cases[case_id] = case
        return case, True

    def get(self, case_id: str) -> ExcursionCase | None:
        return self._cases.get(case_id)
