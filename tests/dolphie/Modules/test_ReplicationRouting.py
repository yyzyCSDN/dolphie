from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from unittest.mock import MagicMock

import pytest

from dolphie.DataTypes import DatabaseRow
from dolphie.Modules.Queries import MySQLQueries
from dolphie.Modules.ReplicationRouting import (
    ChannelFilters,
    RoutingAudit,
    Verdict,
    build_routing_audit_renderable,
    like_match,
    load_routing_audit_config,
    parse_channel_filters,
    route_table,
    run_routing_audit,
)
from dolphie.Widgets.RoutingAuditModal import RoutingAuditModal

_ALL_FILTER_FIELDS = (
    "Replicate_Do_DB",
    "Replicate_Ignore_DB",
    "Replicate_Do_Table",
    "Replicate_Ignore_Table",
    "Replicate_Wild_Do_Table",
    "Replicate_Wild_Ignore_Table",
    "Replicate_Rewrite_DB",
)


def status_row(channel: str = "ch", **values: Any) -> DatabaseRow:
    row: DatabaseRow = {"Channel_Name": channel}
    for field in _ALL_FILTER_FIELDS:
        row[field] = values.get(field, "")
    return row


def make_filters(channel: str = "ch", values: Mapping[str, Any] | None = None) -> ChannelFilters:
    return parse_channel_filters(channel, status_row(channel, **(values or {})))


def route(
    db: str,
    table: str,
    filters: Mapping[str, Any] | None = None,
    *,
    lctn: int | None = 0,
    channel: str = "ch",
):
    return route_table(make_filters(channel, filters), db, table, lctn)


def run_audit(spec: list[tuple[str, dict[str, Any], list[tuple[str, str]]]], lctn: int | None = 0) -> RoutingAudit:
    inputs = [(make_filters(channel, filter_values), tables) for channel, filter_values, tables in spec]
    return run_routing_audit(inputs, lctn)


def collision_targets(audit: RoutingAudit) -> set[tuple[str, str]]:
    return {(collision.target_db, collision.target_table) for collision in audit.collisions}


# --------------------------------------------------------------------------- #
# Wildcard / escape semantics
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("pattern", "text", "expected"),
    [
        ("ord%", "orders", True),
        ("ord_", "ordx", True),
        ("ord_", "orders", False),
        ("%", "anything", True),
        ("_", "x", True),
        ("_", "xy", False),
        ("a%b.c_d", "axxb.c5d", True),
        ("a.b", "a.b", True),  # "." is an ordinary literal character in a LIKE pattern
    ],
)
def test_like_match_percent_underscore(pattern: str, text: str, expected: bool):
    assert like_match(pattern, text, case_sensitive=True) is expected


def test_like_match_escaped_wildcards_are_literal():
    assert like_match(r"100\_save", "100_save", case_sensitive=True)
    assert not like_match(r"100\_save", "100xsave", case_sensitive=True)
    assert like_match(r"take\%5", "take%5", case_sensitive=True)
    assert not like_match(r"take\%5", "takeX5", case_sensitive=True)
    assert like_match(r"back\\slash", "back\\slash", case_sensitive=True)


def test_like_match_dangling_escape_never_matches():
    assert not like_match("orders\\", "orders", case_sensitive=True)
    assert not like_match("orders\\", "orders\\", case_sensitive=True)


def test_like_match_case_sensitivity_follows_flag():
    assert like_match("ORD%", "orders", case_sensitive=False)
    assert not like_match("ORD%", "orders", case_sensitive=True)


# --------------------------------------------------------------------------- #
# ROW filter priority and rule shadowing
# --------------------------------------------------------------------------- #


def test_no_filters_writes_source_table():
    result = route("sales", "orders")

    assert result.verdict is Verdict.write
    assert result.target == ("sales", "orders")
    assert result.traces[0][1].path_tokens == ["default-accept"]


def test_do_db_match_continues_to_table_rules_and_writes():
    result = route("sales", "orders", {"Replicate_Do_DB": "sales"})

    assert result.verdict is Verdict.write
    stages = [step.stage for step in result.traces[0][1].trace]
    assert stages == ["do-db", "decision"]


def test_do_db_miss_excludes_before_any_table_rule_can_rescue():
    # A do-db allowlist that does not contain the affected database excludes the row even
    # though a replicate-wild-do-table rule would match the table.
    result = route("other", "orders", {"Replicate_Do_DB": "sales", "Replicate_Wild_Do_Table": "other.ord%"})

    assert result.verdict is Verdict.excluded
    assert [step.stage for step in result.traces[0][1].trace] == ["do-db"]


def test_ignore_db_match_excludes_and_miss_continues():
    excluded = route("scratch", "t", {"Replicate_Ignore_DB": "scratch"})
    written = route("sales", "t", {"Replicate_Ignore_DB": "scratch"})

    assert excluded.verdict is Verdict.excluded
    assert written.verdict is Verdict.write


def test_database_level_ignore_shadows_an_explicit_do_table():
    # Database rules are evaluated first for ROW events; do-table cannot rescue the table.
    result = route(
        "tmp",
        "keep",
        {"Replicate_Ignore_DB": "tmp", "Replicate_Do_Table": "tmp.keep"},
    )

    assert result.verdict is Verdict.excluded
    assert result.traces[0][1].path_tokens == ["ignore-db:tmp"]


def test_do_table_shadows_ignore_table_and_wild_ignore():
    result = route(
        "a",
        "t",
        {
            "Replicate_Do_Table": "a.t",
            "Replicate_Ignore_Table": "a.t",
            "Replicate_Wild_Ignore_Table": "a.%",
        },
    )

    assert result.verdict is Verdict.write
    assert result.target == ("a", "t")
    assert result.traces[0][1].path_tokens == ["do-table:a.t"]
    assert [step.stage for step in result.traces[0][1].trace] == ["do-table"]


def test_ignore_table_fires_before_wild_do_table():
    result = route(
        "keep",
        "bad",
        {"Replicate_Ignore_Table": "keep.bad", "Replicate_Wild_Do_Table": "keep.%"},
    )

    assert result.verdict is Verdict.excluded
    assert result.traces[0][1].path_tokens == ["ignore-table:keep.bad"]


def test_wild_rules_match_and_take_priority_over_db_agnostic_decision():
    written = route("keep", "orders", {"Replicate_Wild_Do_Table": "keep.ord%"})
    ignored = route("keep", "secret", {"Replicate_Wild_Ignore_Table": "keep.sec%"})

    assert written.verdict is Verdict.write
    assert written.target == ("keep", "orders")
    assert ignored.verdict is Verdict.excluded


def test_do_table_allowlist_catchall_excludes_unmatched_tables():
    result = route("keep", "other")

    assert result.verdict is Verdict.write  # no rules at all

    excluded = route("keep", "other", {"Replicate_Do_Table": "keep.only_this"})
    assert excluded.verdict is Verdict.excluded
    assert excluded.traces[0][1].trace[-1].stage == "decision"


def test_wild_do_allowlist_catchall_excludes_unmatched():
    result = route("keep", "nope", {"Replicate_Wild_Do_Table": "keep.yes%"})

    assert result.verdict is Verdict.excluded
    assert "allowlists exist but none matched" in result.traces[0][1].trace[-1].detail


def test_dotless_wild_pattern_matches_table_in_any_database():
    values = {"Replicate_Wild_Do_Table": "orders"}
    assert route("shop_a", "orders", values).verdict is Verdict.write
    assert route("shop_b", "orders", values).verdict is Verdict.write
    # The catch-all still applies: only a table named exactly "orders" is allowed.
    assert route("shop_a", "orders_archive", values).verdict is Verdict.excluded


# --------------------------------------------------------------------------- #
# Database rewriting
# --------------------------------------------------------------------------- #


def test_rewrite_is_applied_before_database_filters():
    # The rewritten database (dst) is what ignore-db evaluates; the source name is not tested.
    excluded = route("src", "t", {"Replicate_Rewrite_DB": "src->dst", "Replicate_Ignore_DB": "dst"})
    assert excluded.verdict is Verdict.excluded
    assert excluded.traces[0][1].path_tokens == ["rewrite:src->dst", "ignore-db:dst"]

    # Ignoring the original (pre-rewrite) database name does nothing after the rewrite.
    written = route("src", "t", {"Replicate_Rewrite_DB": "src->dst", "Replicate_Ignore_DB": "src"})
    assert written.verdict is Verdict.write
    assert written.target == ("dst", "t")


def test_rewrite_changes_target_database_only():
    result = route("src_a", "orders", {"Replicate_Rewrite_DB": "src_a->merged"})

    assert result.verdict is Verdict.write
    assert result.target == ("merged", "orders")


def test_rewrite_uses_first_matching_mapping_and_does_not_chain():
    result = route(
        "src",
        "t",
        {"Replicate_Rewrite_DB": "src->dst,dst->faraway"},
    )

    assert result.verdict is Verdict.write
    assert result.target == ("dst", "t")  # not faraway: rewrites never chain


# --------------------------------------------------------------------------- #
# Cross-channel collisions
# --------------------------------------------------------------------------- #


def test_post_rewrite_collision_is_confirmed_with_both_rule_paths():
    audit = run_audit(
        [
            ("channel_a", {"Replicate_Rewrite_DB": "src_a->mrg"}, [("src_a", "orders")]),
            ("channel_b", {"Replicate_Rewrite_DB": "src_b->mrg"}, [("src_b", "orders")]),
        ]
    )

    assert len(audit.collisions) == 1
    collision = audit.collisions[0]
    assert collision.confirmed
    assert (collision.target_db, collision.target_table) == ("mrg", "orders")
    assert {side.channel for side in collision.sides} == {"channel_a", "channel_b"}
    # Each side carries its own source table and the full rule path that produced the target.
    assert {side.source_db for side in collision.sides} == {"src_a", "src_b"}
    paths = {side.channel: side.path_tokens for side in collision.sides}
    assert paths["channel_a"] == ("rewrite:src_a->mrg", "default-accept")
    assert paths["channel_b"] == ("rewrite:src_b->mrg", "default-accept")


def test_same_table_name_in_different_databases_is_not_a_collision():
    audit = run_audit(
        [
            ("channel_a", {}, [("db_a", "orders")]),
            ("channel_b", {}, [("db_b", "orders")]),
        ]
    )

    assert audit.collisions == []


def test_rewritten_disjoint_targets_are_not_reported():
    audit = run_audit(
        [
            ("channel_a", {"Replicate_Rewrite_DB": "src_a->dst_a"}, [("src_a", "orders")]),
            ("channel_b", {"Replicate_Rewrite_DB": "src_b->dst_b"}, [("src_b", "orders")]),
        ]
    )

    assert collision_targets(audit) == set()


def test_excluded_routes_never_participate_in_collisions():
    audit = run_audit(
        [
            ("channel_a", {}, [("mrg", "t")]),
            ("channel_b", {"Replicate_Ignore_DB": "mrg"}, [("mrg", "t")]),
        ]
    )

    assert audit.collisions == []
    assert {route.verdict for route in audit.routes} == {Verdict.write, Verdict.excluded}


def test_tables_converging_within_one_channel_are_not_a_collision():
    audit = run_audit(
        [
            (
                "only_channel",
                {"Replicate_Rewrite_DB": "a->mrg,b->mrg"},
                [("a", "t"), ("b", "t")],
            )
        ]
    )

    assert audit.collisions == []


def test_unrelated_tables_are_not_combined():
    audit = run_audit(
        [
            ("channel_a", {}, [("db_a", "customers")]),
            ("channel_b", {}, [("db_b", "orders")]),
        ]
    )

    assert audit.collisions == []


# --------------------------------------------------------------------------- #
# lower_case_table_names semantics
# --------------------------------------------------------------------------- #


def test_case_sensitive_server_keeps_case_variants_distinct():
    audit = run_audit(
        [
            ("channel_a", {"Replicate_Do_DB": "Sales"}, [("Sales", "T")]),
            ("channel_b", {"Replicate_Do_DB": "sales"}, [("sales", "t")]),
        ],
        lctn=0,
    )

    targets = {(route.target_db, route.target_table) for route in audit.routes if route.verdict is Verdict.write}
    assert targets == {("Sales", "T"), ("sales", "t")}
    assert audit.collisions == []

    # A lowercase table against an uppercase do-db is excluded on a case-sensitive server.
    assert route("sales", "t", {"Replicate_Do_DB": "Sales"}, lctn=0).verdict is Verdict.excluded


def test_case_insensitive_server_merges_case_variants_into_confirmed_collision():
    audit = run_audit(
        [
            ("channel_a", {}, [("Sales", "T")]),
            ("channel_b", {}, [("sales", "t")]),
        ],
        lctn=1,
    )

    assert len(audit.collisions) == 1
    assert audit.collisions[0].confirmed
    assert {side.channel for side in audit.collisions[0].sides} == {"channel_a", "channel_b"}


def test_unknown_lctn_still_confirms_exactly_equal_targets():
    audit = run_audit(
        [
            ("channel_a", {"Replicate_Rewrite_DB": "src_a->mrg"}, [("src_a", "orders")]),
            ("channel_b", {"Replicate_Rewrite_DB": "src_b->mrg"}, [("src_b", "orders")]),
        ],
        lctn=None,
    )

    assert len(audit.collisions) == 1
    assert audit.collisions[0].confirmed
    assert all(route.verdict is Verdict.write for route in audit.routes)


def test_unknown_lctn_marks_case_only_cross_channel_targets_undeterminable():
    audit = run_audit(
        [
            ("channel_a", {}, [("Sales", "customers")]),
            ("channel_b", {}, [("sales", "customers")]),
        ],
        lctn=None,
    )

    # Both routes are definite writes on their own; the collision itself cannot be decided.
    assert all(route.verdict is Verdict.write for route in audit.routes)
    assert len(audit.collisions) == 1
    collision = audit.collisions[0]
    assert not collision.confirmed
    assert {side.channel for side in collision.sides} == {"channel_a", "channel_b"}

    # Different databases must not be merged even when case is unknown.
    disjoint = run_audit(
        [("channel_a", {}, [("dbA", "x")]), ("channel_b", {}, [("dbb", "x")])],
        lctn=None,
    )
    assert disjoint.collisions == []


def test_unknown_lctn_makes_case_dependent_filter_route_undeterminable_with_candidate_target():
    result = route("sales", "t", {"Replicate_Do_DB": "Sales"}, lctn=None)

    assert result.verdict is Verdict.undeterminable
    # Case-sensitive server excludes; case-insensitive server writes sales.t, which is reported
    # as a possible target rather than being treated as excluded or unconditionally allowed.
    assert result.possible_targets == {("sales", "t")}
    assert any("case-sensitive" in reason.lower() and "case-insensitive" in reason.lower() for reason in result.reasons)
    assert len(result.traces) == 2


def test_unknown_lctn_divergent_rewrite_targets_are_undeterminable():
    # Case-sensitive: no rewrite matches, the row stays in sales.t. Case-insensitive: the
    # Sales->mrg mapping fires, so it lands in mrg.t. Neither target can be asserted.
    result = route("sales", "t", {"Replicate_Rewrite_DB": "Sales->mrg"}, lctn=None)

    assert result.verdict is Verdict.undeterminable
    assert result.possible_targets == {("sales", "t"), ("mrg", "t")}


def test_possible_write_target_participates_in_undeterminable_collision():
    audit = run_audit(
        [
            ("channel_a", {"Replicate_Rewrite_DB": "Sales->mrg"}, [("sales", "t")]),
            ("channel_b", {}, [("sales", "t")]),
        ],
        lctn=None,
    )

    # channel_a may keep sales.t (matching channel_b's definite target) or diverge to mrg.t;
    # the shared sales.t target cannot be cleared as collision-free.
    assert len(audit.collisions) == 1
    collision = audit.collisions[0]
    assert not collision.confirmed
    assert (collision.target_db, collision.target_table) == ("sales", "t")
    assert {side.channel for side in collision.sides} == {"channel_a", "channel_b"}


# --------------------------------------------------------------------------- #
# Missing privileges / configuration and unsupported rules
# --------------------------------------------------------------------------- #


def test_unreadable_channel_status_is_undeterminable_not_allowed():
    filters = parse_channel_filters("offline", None)

    assert not filters.available
    assert filters.blocked

    result = route_table(filters, "mrg", "t", 0)
    assert result.verdict is Verdict.undeterminable
    assert result.possible_targets == set()
    assert result.target is None
    assert any("REPLICATION CLIENT" in reason for reason in result.reasons)


def test_blocked_channel_neither_confirms_nor_clears_collisions():
    # Simulate a SHOW REPLICA STATUS row that could not be read for the second channel.
    blocked = parse_channel_filters("unreadable", None)
    audit = run_routing_audit(
        [
            (make_filters("good"), [("mrg", "t")]),
            (blocked, [("mrg", "t")]),
        ],
        0,
    )

    verdicts = {route.channel: route.verdict for route in audit.routes}
    assert verdicts == {"good": Verdict.write, "unreadable": Verdict.undeterminable}
    # A possible collision with the unreadable channel must not be confirmed or silently dropped.
    assert audit.collisions == []
    assert any(notice.channel == "unreadable" and "cannot be ruled out" in notice.message for notice in audit.notices)


def test_malformed_rewrite_rule_affects_only_its_channel():
    audit = run_audit(
        [
            ("broken", {"Replicate_Rewrite_DB": "src=>dst"}, [("src", "t")]),
            ("healthy", {}, [("src", "t")]),
        ]
    )

    by_channel = {route.channel: route.verdict for route in audit.routes}
    assert by_channel["broken"] is Verdict.undeterminable
    assert by_channel["healthy"] is Verdict.write
    assert any(notice.channel == "broken" and "Replicate_Rewrite_DB" in notice.message for notice in audit.notices)


def test_malformed_table_rule_marks_channel_undeterminable():
    filters = make_filters("ch", {"Replicate_Do_Table": "no_dot_here"})

    assert filters.blocked
    result = route_table(filters, "db", "no_dot_here", 0)
    assert result.verdict is Verdict.undeterminable


def test_missing_filter_field_marks_channel_undeterminable():
    # A server/version that does not report the Replicate_* fields cannot be evaluated.
    filters = parse_channel_filters("old", {"Channel_Name": "old"})

    assert filters.blocked
    assert any("not reported" in issue for issue in filters.issues)
    assert route_table(filters, "a", "b", 0).verdict is Verdict.undeterminable


def test_unrelated_channels_stay_deterministic_when_one_channel_is_blocked():
    audit = run_routing_audit(
        [
            (parse_channel_filters("blocked", None), [("a", "t")]),
            (make_filters("fine", {"Replicate_Wild_Ignore_Table": "a.%"}), [("a", "t"), ("b", "t")]),
        ],
        0,
    )

    fine_routes = [route for route in audit.routes if route.channel == "fine"]
    assert {(route.source_db, route.verdict) for route in fine_routes} == {
        ("a", Verdict.excluded),
        ("b", Verdict.write),
    }


# --------------------------------------------------------------------------- #
# Read-only configuration loading
# --------------------------------------------------------------------------- #


def test_loader_reads_lctn_and_sorted_base_tables():
    connection = MagicMock()
    connection.fetch_value_from_field.return_value = 1
    connection.last_execute_successful = True
    connection.fetchall.return_value = [
        {"table_schema": "b", "table_name": "z"},
        {"table_schema": "a", "table_name": "y"},
    ]

    config = load_routing_audit_config(connection, [status_row("c1"), status_row("c2")])

    assert config.lower_case_table_names == 1
    assert config.base_tables == [("a", "y"), ("b", "z")]
    assert set(config.filters_by_channel) == {"c1", "c2"}
    executed_query = connection.execute.call_args.args[0]
    assert executed_query.endswith(MySQLQueries.base_tables.strip()) or MySQLQueries.base_tables in executed_query


def test_loader_reports_unreadable_lctn_and_failed_table_list():
    connection = MagicMock()
    connection.fetch_value_from_field.return_value = None

    def failing_execute(*_args, **_kwargs):
        connection.last_execute_successful = False

    connection.execute.side_effect = failing_execute

    config = load_routing_audit_config(connection, [status_row("c1")])

    assert config.lower_case_table_names is None
    assert config.base_tables == []
    messages = "\n".join(config.notices)
    assert "lower_case_table_names" in messages
    assert "information_schema" in messages


# --------------------------------------------------------------------------- #
# Modal input validation and rendered report
# --------------------------------------------------------------------------- #


def test_modal_parses_explicit_table_references():
    assert RoutingAuditModal.parse_table_ref("sales.orders") == ("sales", "orders")
    assert RoutingAuditModal.parse_table_ref(" `sales` . `orders` ") == ("sales", "orders")

    for invalid in ("orders", "a.b.c", ".t", "a.", ""):
        with pytest.raises(ValueError, match="schema.table"):
            RoutingAuditModal.parse_table_ref(invalid)


def test_renderable_contains_verdicts_collision_status_and_scope():
    audit = run_audit(
        [
            ("channel_a", {"Replicate_Rewrite_DB": "src_a->mrg"}, [("src_a", "orders")]),
            ("channel_b", {"Replicate_Ignore_DB": "mrg"}, [("mrg", "orders")]),
            ("channel_c", {"Replicate_Rewrite_DB": "src_c->mrg"}, [("src_c", "orders")]),
        ],
        lctn=0,
    )
    # Force an undeterminable route into the same report to exercise that rendering path.
    audit.routes.append(route_table(parse_channel_filters("dark", None), "mrg", "orders", 0))

    from rich.console import Console

    console = Console(force_terminal=False, no_color=True, width=250)
    with console.capture() as capture:
        console.print(build_routing_audit_renderable(audit))
    text = capture.get()

    assert "CONFIRMED" in text
    assert "EXCLUDED" in text
    assert "UNDETERMINABLE" in text
    assert "channel_a" in text
    assert "channel_b" in text
    assert "rewrite-db" in text
    assert "IGNORE_SERVER_IDS" in text
    assert "lower_case_table_names" in text
