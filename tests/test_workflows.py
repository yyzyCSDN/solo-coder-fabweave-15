from __future__ import annotations

import pytest

from fabweave.chamber.matching import Tolerance, mismatch_reasons
from fabweave.chemicals.shelf_life import effective_age, shelf_life_remaining
from fabweave.cleanroom.excursion import active_excursion, excursion_periods
from fabweave.contamination.segregation import incompatible_pairs, sequence_compatible
from fabweave.core.decision import allow, deny
from fabweave.core.events import EventLog
from fabweave.core.ids import deterministic_id
from fabweave.core.quantities import require_conservation
from fabweave.core.timebox import TimeBox, covered_duration, merge_timeboxes
from fabweave.core.units import fraction_to_ppm, ppm_to_fraction
from fabweave.core.versioning import RevisionHistory
from fabweave.dedication.tool import DedicationPolicy, policy_decision
from fabweave.dispatch.priority import DispatchCandidate, rank_candidates
from fabweave.energy.demand import coincident_peak, demand_violations
from fabweave.genealogy.carrier import GenealogyGraph
from fabweave.holds.release import HoldState, release_decision
from fabweave.lot.merge import Lot, merge_lots
from fabweave.maintenance.window import MaintenancePolicy, assignment_decision
from fabweave.metrology.sampling import Wafer, deterministic_sample
from fabweave.queue_time.enforcer import QueueLimit, queue_time_decision
from fabweave.recipe.qualification import Qualification, qualification_decision
from fabweave.reticle.reservation import Reservation, ReservationBook
from fabweave.rework.authorization import ReworkPolicy, authorization_decision
from fabweave.shipment.commit import CommitLedger
from fabweave.spc.control import western_electric_rules
from fabweave.takt.monitor import takt_result
from fabweave.yielding.excursion import affected_wip, contain
from fabweave.yielding.rollup import StepYield, route_yield
from fabweave.workflows.control import (
    chamber_sample_decision,
    complete_case_disposition,
    containment_scope,
    dedicated_qualification,
    environmental_containment,
    evaluate_sample,
    material_readiness,
    merge_and_commit,
    performance_summary,
    powered_assignment,
    qualified_assignment,
    rank_releasable,
    record_case_retest,
    record_containment,
    release_yield_containment,
    reserve_qualified,
    rework_release_decision,
    start_yield_containment,
)


def test_timeboxes_are_half_open_and_merge_without_double_counting():
    assert not TimeBox(0, 10).overlaps(TimeBox(10, 20))
    assert merge_timeboxes([TimeBox(5, 15), TimeBox(0, 10), TimeBox(20, 25)]) == [TimeBox(0, 15), TimeBox(20, 25)]
    assert covered_duration([TimeBox(-5, 5), TimeBox(3, 9), TimeBox(20, 30)], TimeBox(0, 10)) == 9


def test_event_log_is_idempotent_and_does_not_leak_mutable_data():
    log = EventLog()
    payload = {"items": [1]}
    first = log.append("LOT-1", 0, "EVT-1", 10, "opened", payload)
    payload["items"].append(2)
    replay = log.append("LOT-1", 0, "EVT-1", 10, "opened", {"items": [1]})
    assert first == replay and log.read("LOT-1")[0].data == {"items": [1]}
    with pytest.raises(ValueError):
        log.append("LOT-1", 0, "EVT-1", 10, "opened", {"items": [9]})


def test_event_log_rejects_stale_version_and_backdated_append():
    log = EventLog()
    log.append("LOT-1", 0, "EVT-1", 10, "opened", {})
    with pytest.raises(ValueError):
        log.append("LOT-1", 0, "EVT-2", 11, "moved", {})
    with pytest.raises(ValueError):
        log.append("LOT-1", 1, "EVT-3", 9, "moved", {})


def test_revision_history_uses_effective_time_and_isolated_values():
    history: RevisionHistory[dict[str, int]] = RevisionHistory()
    source = {"limit": 2}
    history.publish(1, 10, source)
    source["limit"] = 99
    history.publish(2, 20, {"limit": 3})
    assert history.at(15).value == {"limit": 2}
    assert history.latest().version == 2


def test_decision_requires_explanation_only_for_rejections():
    assert allow(version=1).allowed
    assert deny("blocked", version=1).reasons == ("blocked",)
    with pytest.raises(ValueError):
        deny()


def test_quantity_conservation_and_deterministic_ids():
    require_conservation({"A": 10, "B": 15}, {"C": 25})
    assert deterministic_id("LOT", "request", 1) == deterministic_id("LOT", "request", 1)
    with pytest.raises(ValueError):
        require_conservation({"A": 10}, {"B": 9})


def test_unit_conversions_validate_ranges_and_round_trip():
    assert fraction_to_ppm(ppm_to_fraction(12.5)) == pytest.approx(12.5)
    with pytest.raises(ValueError):
        ppm_to_fraction(-1)
    with pytest.raises(ValueError):
        fraction_to_ppm(1.01)


def test_queue_time_clips_and_merges_authorized_pauses():
    decision = queue_time_decision(
        QueueLimit("ETCH", 80, 3),
        100,
        200,
        [TimeBox(80, 110), TimeBox(105, 130), TimeBox(190, 220)],
    )
    assert decision.allowed
    assert decision.evidence["paused_minutes"] == 40
    assert decision.evidence["remaining_minutes"] == 20


def test_dispatch_excludes_held_and_frozen_lots_and_breaks_ties_stably():
    candidates = [
        DispatchCandidate("LOT-B", 1, 2),
        DispatchCandidate("LOT-A", 1, 2),
        DispatchCandidate("LOT-H", 0.1, 1, active_hold=True),
        DispatchCandidate("LOT-F", 0.1, 1, frozen_until=20),
    ]
    assert [item.lot_id for item in rank_candidates(candidates, 10)] == ["LOT-A", "LOT-B"]


def test_reticle_batch_reservation_is_atomic_and_idempotent():
    book = ReservationBook()
    initial = Reservation("RET-1", "JOB-1", TimeBox(0, 10), 1)
    assert book.reserve_batch("REQ-1", [initial]) == (initial,)
    assert book.reserve_batch("REQ-1", [initial]) == (initial,)
    with pytest.raises(ValueError):
        book.reserve_batch(
            "REQ-2",
            [Reservation("RET-2", "JOB-2", TimeBox(0, 10), 1), Reservation("RET-1", "JOB-3", TimeBox(5, 15), 1)],
        )
    assert book.reservations("RET-2") == ()


def test_qualification_selection_honors_chamber_revision_expiry_and_revocation():
    qualifications = [
        Qualification("TOOL-1", "CH-1", 2, 0, 100),
        Qualification("TOOL-1", "CH-2", 2, 0, 100),
        Qualification("TOOL-1", "CH-1", 3, 0, 100, revoked_at=40),
    ]
    assert qualification_decision("TOOL-1", "CH-1", 2, 50, qualifications).allowed
    assert not qualification_decision("TOOL-1", "CH-1", 3, 50, qualifications).allowed


@pytest.mark.parametrize(
    "values,rule",
    [([0, 0, 3.1], "one_beyond_3_sigma"), ([2.1, 2.2, 0], "two_of_three_beyond_2_sigma_same_side"), ([1.1, 1.2, 0, 1.3, 1.4], "four_of_five_beyond_1_sigma_same_side")],
)
def test_spc_rules(values, rule):
    assert rule in western_electric_rules(values, 0, 1)


def test_metrology_sample_covers_required_zones_and_is_deterministic():
    wafers = [Wafer(f"W{i}", i, "EDGE" if i in {0, 5} else "CENTER") for i in range(6)]
    first = deterministic_sample(wafers, 3, {"EDGE", "CENTER"})
    assert first == deterministic_sample(list(reversed(wafers)), 3, {"EDGE", "CENTER"})
    assert {wafer.zone for wafer in first} == {"EDGE", "CENTER"}


def test_maintenance_assignment_combines_both_clocks_and_planned_window():
    policy = MaintenancePolicy(2, run_interval=100, calendar_interval=50)
    assert "calendar_maintenance_due" in assignment_decision(policy, 0, 10, 60, TimeBox(70, 80), []).reasons
    assert "planned_maintenance_overlap" in assignment_decision(policy, 50, 10, 60, TimeBox(70, 80), [TimeBox(75, 90)]).reasons


def test_chamber_matching_rejects_missing_non_finite_and_outside_parameters():
    tolerances = {"pressure": Tolerance(absolute=1), "flow": Tolerance(relative=0.1)}
    reasons = mismatch_reasons({"pressure": 10, "flow": 100}, {"pressure": 12}, tolerances)
    assert reasons == ("flow:missing", "pressure:outside_tolerance")


def test_hold_release_rejects_stale_version_missing_disposition_and_blockers():
    state = HoldState("H-1", 3, frozenset({"QA"}), frozenset(), frozenset({"CUSTOMER"}))
    assert release_decision(state, 2, 100).reasons == ("stale_hold_version",)
    reasons = release_decision(state, 3, 100).reasons
    assert reasons == ("missing_disposition:QA", "active_blocker:CUSTOMER")


def test_genealogy_traverses_both_directions_without_duplicates():
    graph = GenealogyGraph({"B": "A", "C": "A", "D": "B", "A": None})
    assert graph.related("D", "upstream", 5) == ("A", "B")
    assert graph.related("A", "downstream", 5) == ("B", "C", "D")
    with pytest.raises(ValueError):
        GenealogyGraph({"A": "B", "B": "A"})


def test_chemical_age_merges_and_clips_freeze_periods():
    periods = [TimeBox(-5, 5), TimeBox(3, 8), TimeBox(20, 30)]
    assert effective_age(0, 10, periods) == 2
    assert shelf_life_remaining(0, 5, 10, periods) == 3


def test_cleanroom_event_time_dedup_and_hysteresis():
    assert excursion_periods([(2, 6), (1, 5), (2, 6), (3, 1)], 4, 2) == [(1, 3)]
    assert active_excursion([(1, 6), (2, 6), (3, 3), (4, 6)], 5, 4, 2, 2)
    with pytest.raises(ValueError):
        excursion_periods([(1, 5), (1, 6)], 4, 1)


def test_coincident_demand_groups_timestamps_and_uses_half_open_window():
    loads = [(0, 2), (5, 3), (5, 4), (10, 7)]
    assert coincident_peak(loads, 10) == 14
    assert demand_violations(loads, [(5, 10), (10, 13)]) == (10,)


def test_rework_authorization_combines_budget_defect_and_engineer_gate():
    policy = ReworkPolicy(4, {"ETCH": 2}, frozenset({"D1", "D2"}), frozenset({"D2"}))
    assert authorization_decision(policy, "ETCH", 1, "D1", None).allowed
    assert authorization_decision(policy, "ETCH", 2, "D2", None).reasons == (
        "rework_limit_exhausted",
        "engineer_approval_missing",
    )


def test_lot_merge_preserves_quantity_and_rejects_held_input():
    lots = [Lot("LOT-A", "P", "R", "S", 1, 10), Lot("LOT-B", "P", "R", "S", 1, 15)]
    merged = merge_lots("REQ-1", lots)
    assert merged.quantity == 25
    with pytest.raises(ValueError):
        merge_lots("REQ-2", [lots[0], Lot("LOT-C", "P", "R", "S", 1, 2, frozenset({"QA"}))])


def test_dedication_uses_effective_revision_and_deny_overrides_allow():
    policies = [
        DedicationPolicy(1, 0, {"P": frozenset({"T1", "T2"})}),
        DedicationPolicy(2, 10, {"P": frozenset({"T1", "T2"})}, frozenset({("P", "T2")})),
    ]
    assert policy_decision("P", "T2", 5, policies).allowed
    assert policy_decision("P", "T2", 15, policies).reasons == ("product_tool_explicitly_blocked",)


def test_contamination_matrix_is_symmetric_and_cleaning_changes_residuals():
    matrix = {"CU": {"FE"}}
    assert incompatible_pairs({"CU", "FE"}, matrix) == (("CU", "FE"),)
    assert sequence_compatible({"CU"}, {"FE"}, matrix, cleaning_removes={"CU"})


def test_takt_uses_intervals_and_reports_duplicate_completions():
    result = takt_result([0, 10, 10, 20], 0, 20)
    assert result.takt == 10 and result.interval_count == 2 and result.duplicate_count == 1


def test_route_yield_is_ordered_multiplicative_and_complete():
    steps = [StepYield("B", 100, 80), StepYield("A", 100, 90)]
    assert route_yield(steps, ["A", "B"]) == pytest.approx(0.72)
    with pytest.raises(ValueError):
        route_yield(steps[:1], ["A", "B"])


def test_commit_ledger_is_atomic_idempotent_and_excludes_unavailable_units():
    ledger = CommitLedger(100, safety_units=5)
    ledger.mark_unavailable(pending_scrap=10, quarantine=5)
    first = ledger.allocate("REQ-1", "ORDER-1", 50)
    assert ledger.allocate("REQ-1", "ORDER-1", 50) == first
    assert ledger.available == 30
    with pytest.raises(ValueError):
        ledger.allocate("REQ-2", "ORDER-2", 31)
    assert ledger.available == 30


def test_material_readiness_combines_queue_and_chemical_clocks():
    decision = material_readiness(QueueLimit("ETCH", 80, 2), 100, 200, [TimeBox(110, 130)], 100, 90, [TimeBox(110, 130)])
    assert decision.allowed and decision.evidence["remaining_minutes"] == 0 and decision.evidence["remaining_shelf_hours"] == 10


def test_rank_releasable_applies_hold_decisions_before_dispatch():
    candidates = [DispatchCandidate("A", 1, 2), DispatchCandidate("B", 2, 2)]
    holds = {"A": HoldState("H", 1, frozenset({"QA"}))}
    assert [item.lot_id for item in rank_releasable(candidates, 10, holds, {"A": 1})] == ["B"]


def test_reserve_qualified_is_fail_closed_and_then_atomic():
    book = ReservationBook()
    reservation = Reservation("R", "J", TimeBox(10, 20), 3)
    qualifications = [Qualification("T", "C", 3, 0, 30)]
    assert reserve_qualified(book, "REQ", [reservation], {"J": ("T", "C")}, qualifications, 10).allowed
    assert len(book.reservations("R")) == 1


def test_qualified_assignment_combines_maintenance_and_recipe_state():
    decision = qualified_assignment(MaintenancePolicy(1, 100, 100), 0, 1, 10, TimeBox(20, 30), [], "T", "C", 2, [Qualification("T", "C", 2, 0, 40)])
    assert decision.allowed and decision.evidence["recipe_revision"] == 2


def test_sample_evaluation_keeps_selected_membership_with_spc_rules():
    wafers = [Wafer(f"W{i}", i, "EDGE" if i in {0, 4} else "CENTER") for i in range(5)]
    result = evaluate_sample(wafers, 3, {"EDGE", "CENTER"}, {f"W{i}": value for i, value in enumerate([0, 0, 0, 0, 4])}, 0, 1)
    assert len(result.wafer_ids) == 3 and "one_beyond_3_sigma" in result.rules


def test_chamber_sample_reports_parameter_and_coverage_failures_together():
    decision = chamber_sample_decision({"p": 10}, {"p": 12}, {"p": Tolerance(absolute=1)}, [Wafer("W", 0, "CENTER")], 1, {"EDGE"})
    assert decision.reasons[0] == "p:outside_tolerance" and decision.reasons[1].startswith("sample:")


def test_environmental_containment_combines_excursion_and_material_risk():
    decision = environmental_containment([(1, 6), (2, 6)], 5, 4, 2, 2, {"CU", "FE"}, {"CU": {"FE"}})
    assert decision.reasons == ("cleanroom_excursion", "incompatible:CU:FE")


def test_containment_scope_is_bounded_bidirectional_and_unique():
    graph = GenealogyGraph({"A": None, "B": "A", "C": "B", "D": "A"})
    assert containment_scope(graph, ["B"], 1) == ("A", "B", "C")


def test_containment_record_is_idempotent_and_binds_the_planned_scope():
    graph = GenealogyGraph({"A": None, "B": "A", "C": "B"})
    log = EventLog()
    first = record_containment(log, "RECALL-1", graph, ["B"], 1, 10)
    replay = record_containment(log, "RECALL-1", graph, ["B"], 1, 10)
    assert first == replay and first.scope == ("A", "B", "C") and first.event_sequence == 1


def test_powered_assignment_combines_peak_and_maintenance_reasons():
    decision = powered_assignment([(0, 7), (1, 7)], [(5, 10)], MaintenancePolicy(1, 100, 5), 0, 1, 10, TimeBox(10, 20), [])
    assert decision.reasons == ("demand_limit_exceeded:5", "calendar_maintenance_due")


def test_rework_release_requires_both_authorization_and_hold_completion():
    policy = ReworkPolicy(1, {"ETCH": 2}, frozenset({"D"}))
    hold = HoldState("H", 1, frozenset({"QA"}), frozenset())
    assert rework_release_decision(policy, "ETCH", 0, "D", None, hold, 1, 10).reasons == ("missing_disposition:QA",)


def test_merge_and_commit_preserves_quantity_across_services():
    lots = [Lot("A", "P", "R", "S", 1, 4), Lot("B", "P", "R", "S", 1, 6)]
    merged, allocation = merge_and_commit("REQ", lots, CommitLedger(20), "ORDER")
    assert merged.quantity == allocation.units == 10


def test_dedicated_qualification_combines_both_policy_gates():
    decision = dedicated_qualification("P", "T", "C", 2, 10, [DedicationPolicy(1, 0, {"P": frozenset({"T"})})], [Qualification("T", "C", 2, 0, 20)])
    assert decision.allowed and decision.evidence["policy_revision"] == 1


def test_performance_summary_uses_filtered_takt_and_ordered_route_yield():
    summary = performance_summary([0, 10, 20], 5, 20, [StepYield("A", 10, 9), StepYield("B", 9, 8)], ["A", "B"])
    assert summary.takt.takt == 10 and summary.route_yield == pytest.approx(0.8)


def _yield_setup(positions, required=("QA",)):
    graph = GenealogyGraph({lot: None for lot in positions})
    log = EventLog()
    rules = western_electric_rules([0, 0, 3.1], 0, 1)
    start = start_yield_containment(log, "YC-1", ["ETCH", "DEP"], "ETCH", positions, graph, 1, set(required), rules, 100)
    return log, start


def test_yield_scope_circles_downstream_wip_and_genealogy_relatives_only():
    graph = GenealogyGraph({"LOT-A": None, "LOT-B": "LOT-A", "LOT-C": "LOT-B", "LOT-D": None, "LOT-X": "LOT-D"})
    route = ["LITHO", "ETCH", "DEP", "CMP"]
    positions = {"LOT-A": "ETCH", "LOT-B": "DEP", "LOT-C": "LITHO", "LOT-D": "CMP"}
    assert affected_wip(route, "ETCH", positions, graph, 2) == ("LOT-A", "LOT-B", "LOT-C", "LOT-D")
    assert affected_wip(route, "CMP", positions, graph, 2) == ()


def test_yield_scope_rejects_empty_duplicate_and_unknown_steps():
    graph = GenealogyGraph({"LOT-A": None})
    with pytest.raises(ValueError):
        affected_wip([], "ETCH", {}, graph, 1)
    with pytest.raises(ValueError):
        affected_wip(["ETCH", "ETCH"], "ETCH", {}, graph, 1)
    with pytest.raises(ValueError):
        affected_wip(["ETCH"], "DEP", {}, graph, 1)
    with pytest.raises(ValueError):
        affected_wip(["ETCH"], "ETCH", {"LOT-A": "DEP"}, graph, 1)


def test_yield_containment_start_is_idempotent_and_scope_cannot_drift():
    graph = GenealogyGraph({"LOT-A": None, "LOT-B": "LOT-A"})
    route = ["LITHO", "ETCH", "DEP"]
    positions = {"LOT-A": "DEP", "LOT-B": "ETCH"}
    log = EventLog()
    rules = western_electric_rules([0, 0, 3.1], 0, 1)
    first = start_yield_containment(log, "YC-1", route, "LITHO", positions, graph, 1, {"QA"}, rules, 100)
    replay = start_yield_containment(log, "YC-1", route, "LITHO", positions, graph, 1, {"QA"}, rules, 100)
    assert first == replay
    assert first.case.scope == ("LOT-A", "LOT-B")
    assert [hold.hold_id for hold in first.holds] == ["YC-1:LOT-A", "YC-1:LOT-B"]
    with pytest.raises(ValueError):
        start_yield_containment(log, "YC-1", route, "ETCH", positions, graph, 1, {"QA"}, rules, 100)
    with pytest.raises(ValueError):
        start_yield_containment(log, "YC-2", route, "LITHO", positions, graph, 1, {"QA"}, (), 100)


def test_yield_case_events_cannot_expand_shrink_or_regress_state():
    log, start = _yield_setup({"LOT-A": "DEP"}, required=("QA", "ENG"))
    case = start.case
    assert contain(case, ("LOT-A", "LOT-B"), 101) == case
    assert contain(case, (), 101) == case
    done = complete_case_disposition(log, case, "EVT-QA", "QA", 110).case
    assert done.completed_dispositions == frozenset({"QA"}) and done.version == 3
    assert complete_case_disposition(log, done, "EVT-QA", "QA", 110).case == done
    assert complete_case_disposition(log, done, "EVT-LATE", "ENG", 105).case == done
    assert record_case_retest(log, done, "EVT-RT-LATE", True, 105).case == done
    recovered = record_case_retest(log, done, "EVT-RT", True, 120).case
    assert recovered.retest_recovered
    assert record_case_retest(log, recovered, "EVT-RT-2", False, 130).case == recovered


def test_yield_release_requires_completed_dispositions_and_retest_recovery():
    log, start = _yield_setup({"LOT-A": "DEP", "LOT-B": "DEP"})
    case = start.case
    hold = start.holds[0]
    assert release_decision(hold, hold.version, 110).reasons == ("missing_disposition:QA", "active_blocker:retest_pending")
    _, denied = release_yield_containment(log, case, "EVT-REL", case.version, 110)
    assert denied.reasons == ("missing_disposition:QA", "retest_not_recovered")
    assert len(log.read("yieldcase:YC-1")) == 1
    case = complete_case_disposition(log, case, "EVT-QA", "QA", 120).case
    _, denied = release_yield_containment(log, case, "EVT-REL-2", case.version, 125)
    assert denied.reasons == ("retest_not_recovered",)
    case = record_case_retest(log, case, "EVT-RT", True, 130).case
    with pytest.raises(ValueError):
        complete_case_disposition(log, case, "EVT-X", "UNKNOWN", 140)
    released, decision = release_yield_containment(log, case, "EVT-REL-3", case.version, 140)
    assert decision.allowed and released.case.released_at == 140
    replay, decision = release_yield_containment(log, released.case, "EVT-REL-3", case.version, 140)
    assert decision.allowed and decision.evidence["replay"] and replay == released
    _, stale = release_yield_containment(log, released.case, "EVT-REL-4", 1, 150)
    assert stale.reasons == ("stale_case_version",)
    assert record_case_retest(log, released.case, "EVT-RT-2", False, 150).case == released.case
    assert complete_case_disposition(log, released.case, "EVT-QA-2", "QA", 150).case == released.case


def test_yield_containment_holds_gate_dispatch_until_release():
    log, start = _yield_setup({"LOT-A": "DEP", "LOT-B": "DEP"})
    candidates = [DispatchCandidate("LOT-C", 1, 2), DispatchCandidate("LOT-A", 1, 2), DispatchCandidate("LOT-B", 1, 2)]

    def hold_map(containment):
        return dict(zip(containment.case.scope, containment.holds))

    ranked = rank_releasable(candidates, 110, hold_map(start), {})
    assert [item.lot_id for item in ranked] == ["LOT-C"]
    case = complete_case_disposition(log, start.case, "EVT-QA", "QA", 120).case
    case = record_case_retest(log, case, "EVT-RT", True, 130).case
    released, decision = release_yield_containment(log, case, "EVT-REL", case.version, 140)
    assert decision.allowed
    ranked = rank_releasable(candidates, 150, hold_map(released), {})
    assert [item.lot_id for item in ranked] == ["LOT-A", "LOT-B", "LOT-C"]
