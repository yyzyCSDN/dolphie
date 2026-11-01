"""Version-scoped query rule history: identity, breakpoints, evidence, replay serialization."""

from __future__ import annotations

from typing import Any

from dolphie.Modules.ProxySQLQueryRuleTracker import (
    BREAKPOINT_FIRST,
    BREAKPOINT_RECONNECT,
    BREAKPOINT_RELOAD,
    BREAKPOINT_ROLLBACK,
    BREAKPOINT_UNCONFIRMED,
    QueryRuleTracker,
    RuleSample,
)


def rule(rule_id: int, hits: int, match_pattern: str = ".*", **extra: Any) -> dict[str, Any]:
    row = {
        "rule_id": rule_id,
        "hits": hits,
        "hits_s": hits,
        "active": 1,
        "apply": 1,
        "log": 0,
        "match_pattern": match_pattern,
    }
    row.update(extra)
    return row


def observe(
    tracker: QueryRuleTracker,
    rules: list[dict[str, Any]],
    *,
    at: float,
    runtime_version: int = 1,
    epoch: int = 1000,
    checksum: str = "0xAAAA",
    generation: int = 1,
    confirmed: bool = True,
) -> RuleSample:
    return tracker.record(
        rules,
        runtime_version=runtime_version,
        epoch=epoch,
        checksum=checksum,
        connection_generation=generation,
        confirmed=confirmed,
        observed_at=at,
    )


def test_the_first_sample_only_sets_a_baseline() -> None:
    tracker = QueryRuleTracker()
    sample = observe(tracker, [rule(1, 5)], at=0.0)

    assert sample.breakpoint == BREAKPOINT_FIRST
    assert sample.version_seq == 1
    assert sample.deltas == {"1": None}
    assert sample.rate_for("1") is None
    assert sample.confirmed is True


def test_deltas_are_scoped_to_one_confirmed_version() -> None:
    tracker = QueryRuleTracker()
    observe(tracker, [rule(1, 10), rule(2, 20)], at=0.0)
    sample = observe(tracker, [rule(1, 30), rule(2, 25)], at=2.0)

    assert sample.breakpoint is None
    assert sample.version_seq == 1
    assert sample.deltas == {"1": 20, "2": 5}
    assert sample.rate_for("1") == 10
    assert sample.rate_for("2") == 2  # 2.5 rounds to the nearest even integer


def test_a_reload_starts_a_new_version_even_with_identical_text() -> None:
    tracker = QueryRuleTracker()
    rules = [rule(1, 10, match_pattern="^SELECT")]
    first = observe(tracker, rules, at=0.0)
    # A LOAD restarts the hit counters; the rule text is byte-identical
    reloaded = observe(
        tracker,
        [rule(1, 0, match_pattern="^SELECT")],
        at=1.0,
        runtime_version=2,
        epoch=2000,
        checksum="0xAAAA",
    )

    assert reloaded.breakpoint == BREAKPOINT_RELOAD
    assert reloaded.version_seq == 2
    assert first.version_seq == 1
    assert reloaded.deltas == {"1": None}
    assert reloaded.rate_for("1") is None

    # The identical checksum must not merge the two loads into one identity
    identities = list(tracker.evidence)
    assert len(identities) == 2
    assert identities[0] != identities[1]

    # Rates resume inside the new version instead of reaching back across the reload
    settled = observe(
        tracker,
        [rule(1, 4, match_pattern="^SELECT")],
        at=3.0,
        runtime_version=2,
        epoch=2000,
        checksum="0xAAAA",
    )
    assert settled.breakpoint is None
    assert settled.deltas == {"1": 4}
    assert settled.rate_for("1") == 2


def test_same_rule_id_with_different_text_is_kept_as_separate_evidence() -> None:
    tracker = QueryRuleTracker()
    observe(tracker, [rule(1, 1, match_pattern="^SELECT")], at=0.0)
    observe(
        tracker,
        [rule(1, 0, match_pattern="^INSERT")],
        at=1.0,
        runtime_version=2,
        epoch=2000,
        checksum="0xBBBB",
    )

    identities = list(tracker.evidence)
    assert tracker.evidence[identities[0]][0]["match_pattern"] == "^SELECT"
    assert tracker.evidence[identities[1]][0]["match_pattern"] == "^INSERT"


def test_rule_deletion_and_id_reuse_never_rewrites_old_evidence() -> None:
    tracker = QueryRuleTracker()
    observe(tracker, [rule(1, 5), rule(2, 7, match_pattern="OLD")], at=0.0)
    # rule 2 deleted by a reload
    observe(
        tracker,
        [rule(1, 0)],
        at=1.0,
        runtime_version=2,
        epoch=2000,
        checksum="0xBBBB",
    )
    # rule_id 2 comes back with unrelated text in yet another version
    observe(
        tracker,
        [rule(1, 2), rule(2, 0, match_pattern="NEW")],
        at=2.0,
        runtime_version=3,
        epoch=3000,
        checksum="0xCCCC",
    )

    first_identity = next(identity for identity in tracker.evidence if tracker.evidence[identity][0]["hits"] == 5)
    old_rule_two = next(row for row in tracker.evidence[first_identity] if row["rule_id"] == 2)
    assert old_rule_two["match_pattern"] == "OLD"
    assert len(tracker.evidence) == 3


def test_a_counter_rollback_breaks_only_the_affected_rules() -> None:
    tracker = QueryRuleTracker()
    observe(tracker, [rule(1, 10), rule(2, 10)], at=0.0)
    sample = observe(tracker, [rule(1, 8), rule(2, 14)], at=2.0)

    assert sample.breakpoint == BREAKPOINT_ROLLBACK
    assert sample.rolled_back_rules == ["1"]
    assert sample.deltas == {"1": None, "2": 4}
    assert sample.rate_for("1") is None
    assert sample.rate_for("2") == 2


def test_an_admin_connection_rebuild_is_a_breakpoint_without_a_reload() -> None:
    tracker = QueryRuleTracker()
    observe(tracker, [rule(1, 10)], at=0.0)
    rebuilt = observe(
        tracker,
        [rule(1, 10)],
        at=1.0,
        runtime_version=1,
        epoch=1000,
        checksum="0xAAAA",
        generation=2,
    )

    assert rebuilt.breakpoint == BREAKPOINT_RECONNECT
    assert rebuilt.version_seq == 1  # same runtime identity lineage is not claimed as a new version
    assert rebuilt.deltas == {"1": None}
    assert rebuilt.rate_for("1") is None

    resumed = observe(
        tracker,
        [rule(1, 14)],
        at=3.0,
        runtime_version=1,
        epoch=1000,
        checksum="0xAAAA",
        generation=2,
    )
    assert resumed.breakpoint is None
    assert resumed.deltas == {"1": 4}


def test_unconfirmed_samples_break_the_series_on_both_sides() -> None:
    tracker = QueryRuleTracker()
    observe(tracker, [rule(1, 10)], at=0.0)
    # A reload may have happened between the bracketing reads: unconfirmable
    gap = observe(
        tracker,
        [rule(1, 0)],
        at=1.0,
        runtime_version=2,
        epoch=2000,
        checksum="0xBBBB",
        confirmed=False,
    )
    assert gap.identity is None
    assert gap.version_seq is None
    assert gap.breakpoint == BREAKPOINT_UNCONFIRMED
    assert gap.deltas == {"1": None}

    # The first confirmed sample after the gap cannot be linked through it either
    resumed = observe(
        tracker,
        [rule(1, 2)],
        at=2.0,
        runtime_version=2,
        epoch=2000,
        checksum="0xBBBB",
    )
    assert resumed.breakpoint == BREAKPOINT_UNCONFIRMED
    assert resumed.deltas == {"1": None}

    # The next sample of the same version links normally
    linked = observe(
        tracker,
        [rule(1, 6)],
        at=4.0,
        runtime_version=2,
        epoch=2000,
        checksum="0xBBBB",
    )
    assert linked.breakpoint is None
    assert linked.deltas == {"1": 4}
    assert linked.rate_for("1") == 2


def test_a_new_rule_id_within_a_version_gets_a_baseline_not_a_crossed_rate() -> None:
    tracker = QueryRuleTracker()
    observe(tracker, [rule(1, 10)], at=0.0)
    observe(tracker, [rule(1, 14)], at=1.0)
    added = observe(
        tracker,
        [rule(1, 18), rule(3, 1)],
        at=2.0,
    )
    assert added.deltas == {"1": 4, "3": None}
    assert added.rate_for("3") is None


def test_replay_round_trip_restores_identity_attribution_and_markers() -> None:
    tracker = QueryRuleTracker()
    observe(tracker, [rule(1, 10)], at=0.0)
    original = observe(
        tracker,
        [rule(1, 14)],
        at=2.0,
        runtime_version=2,
        epoch=2000,
        checksum="0xBBBB",
    )

    payload = original.to_replay()
    restored = RuleSample.from_replay(payload)
    assert restored is not None
    assert restored.identity == original.identity
    assert restored.version_seq == original.version_seq
    assert restored.deltas == original.deltas
    assert restored.interval == original.interval
    assert restored.confirmed == original.confirmed
    assert restored.rules == original.rules


def test_replay_parses_absent_or_foreign_payloads_as_unavailable() -> None:
    assert RuleSample.from_replay(None) is None
    assert RuleSample.from_replay([]) is None  # type: ignore[arg-type]

    # An identity with no checksum is treated as no identity
    sample = RuleSample.from_replay(
        {
            "confirmed": True,
            "interval": 1.0,
            "identity": {"runtime_version": 2, "epoch": 2000, "checksum": ""},
            "rules": [{"rule_id": "1", "hits": 3}],
            "deltas": {"1": None},
        }
    )
    assert sample is not None
    assert sample.identity is None
    assert sample.rules == [{"rule_id": "1", "hits": 3}]
