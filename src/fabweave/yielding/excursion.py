from __future__ import annotations

from dataclasses import dataclass, replace

from fabweave.core.decision import Decision, allow, deny
from fabweave.genealogy.carrier import GenealogyGraph
from fabweave.holds.release import HoldState


CASE_STATUSES = ("open", "contained", "released")
RETEST_BLOCKER = "retest_pending"


def affected_wip(
    route: list[str],
    good_step: str,
    positions: dict[str, str],
    graph: GenealogyGraph,
    max_depth: int,
) -> tuple[str, ...]:
    """Circle the work-in-progress still affected by a yield anomaly.

    Seeds are lots positioned strictly downstream of the last confirmed good
    step; genealogy relatives of the seeds are added while they remain in fab.
    """
    if not route:
        raise ValueError("route must contain at least one step")
    if len(set(route)) != len(route):
        raise ValueError("route steps must be unique")
    if good_step not in route:
        raise ValueError("last confirmed good step is not on the route")
    unknown = sorted(set(positions.values()) - set(route))
    if unknown:
        raise ValueError(f"wip positioned at unknown steps: {', '.join(unknown)}")
    good_index = route.index(good_step)
    seeds = sorted(lot for lot, step in positions.items() if route.index(step) > good_index)
    impacted = set(seeds)
    for seed in seeds:
        impacted.update(graph.related(seed, "upstream", max_depth))
        impacted.update(graph.related(seed, "downstream", max_depth))
    return tuple(sorted(impacted & set(positions)))


@dataclass(frozen=True, slots=True)
class YieldCase:
    """Monotonic containment case for one reported yield anomaly.

    Transitions only move forward: the scope is fixed at containment and
    duplicate or late events never expand, shrink or regress the case.
    """

    case_id: str
    good_step: str
    rules: tuple[str, ...]
    status: str
    scope: tuple[str, ...]
    required_dispositions: frozenset[str]
    completed_dispositions: frozenset[str]
    retest_recovered: bool
    updated_at: int
    version: int
    released_at: int | None = None

    def __post_init__(self) -> None:
        if self.status not in CASE_STATUSES:
            raise ValueError("unknown case status")
        if (self.status == "released") != (self.released_at is not None):
            raise ValueError("released status and released_at must agree")
        if self.status == "open" and self.scope:
            raise ValueError("an open case cannot have a scope")
        if tuple(sorted(self.scope)) != self.scope or len(set(self.scope)) != len(self.scope):
            raise ValueError("case scope must be sorted and unique")
        if not self.completed_dispositions <= self.required_dispositions:
            raise ValueError("completed dispositions must be required")
        if self.version <= 0:
            raise ValueError("version must be positive")


def open_case(
    case_id: str,
    good_step: str,
    rules: tuple[str, ...],
    required_dispositions: frozenset[str],
    at: int,
) -> YieldCase:
    if not rules:
        raise ValueError("an SPC anomaly is required to open a yield case")
    return YieldCase(case_id, good_step, tuple(rules), "open", (), frozenset(required_dispositions), frozenset(), False, at, 1)


def contain(case: YieldCase, scope: tuple[str, ...], at: int) -> YieldCase:
    if case.status != "open" or at < case.updated_at:
        return case
    return replace(case, status="contained", scope=tuple(sorted(set(scope))), updated_at=at, version=case.version + 1)


def complete_disposition(case: YieldCase, name: str, at: int) -> YieldCase:
    if case.status == "released" or at < case.updated_at or name in case.completed_dispositions:
        return case
    if case.status != "contained":
        raise ValueError("case is not contained")
    if name not in case.required_dispositions:
        raise ValueError(f"unknown disposition: {name}")
    return replace(case, completed_dispositions=case.completed_dispositions | {name}, updated_at=at, version=case.version + 1)


def record_retest(case: YieldCase, recovered: bool, at: int) -> YieldCase:
    if case.status == "released" or at < case.updated_at:
        return case
    if case.status != "contained":
        raise ValueError("case is not contained")
    if not recovered or case.retest_recovered:
        return case
    return replace(case, retest_recovered=True, updated_at=at, version=case.version + 1)


def case_release_decision(case: YieldCase, expected_version: int, at: int) -> Decision:
    if expected_version != case.version:
        return deny("stale_case_version", current_version=case.version)
    if case.released_at is not None:
        return allow(case_version=case.version, released_at=case.released_at, replay=True)
    reasons = ["case_not_contained"] if case.status != "contained" else []
    reasons.extend(f"missing_disposition:{item}" for item in sorted(case.required_dispositions - case.completed_dispositions))
    if not case.retest_recovered:
        reasons.append("retest_not_recovered")
    if at < case.updated_at:
        reasons.append("release_time_before_last_update")
    evidence = {"case_version": case.version, "evaluated_at": at}
    return deny(*reasons, **evidence) if reasons else allow(**evidence)


def release_case(case: YieldCase, expected_version: int, at: int) -> tuple[YieldCase, Decision]:
    decision = case_release_decision(case, expected_version, at)
    if not decision.allowed or case.released_at is not None:
        return case, decision
    return replace(case, status="released", released_at=at, updated_at=at), decision


def case_hold(case: YieldCase, lot_id: str) -> HoldState:
    if lot_id not in case.scope:
        raise ValueError(f"lot outside case scope: {lot_id}")
    return HoldState(
        hold_id=f"{case.case_id}:{lot_id}",
        version=case.version,
        required_dispositions=case.required_dispositions,
        completed_dispositions=case.completed_dispositions,
        active_blockers=frozenset() if case.retest_recovered else frozenset({RETEST_BLOCKER}),
        released_at=case.released_at,
    )


def case_holds(case: YieldCase) -> tuple[HoldState, ...]:
    return tuple(case_hold(case, lot_id) for lot_id in case.scope)
