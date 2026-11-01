"""The ProxySQL 3.0 query-rules fetch brackets rule reads with runtime_checksums_values."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

from dolphie.Modules.ProxySQLQueryRuleTracker import (
    BREAKPOINT_FIRST,
    BREAKPOINT_UNCONFIRMED,
    QueryRuleTracker,
)
from dolphie.Modules.WorkerDataProcessor import WorkerDataProcessor

VERSION_QUERY_MARKER = "runtime_checksums_values"


class FakeConnection:
    def __init__(self, version_rows: list[list[dict[str, Any]]], rules: list[dict[str, Any]]):
        self.generation = 1
        self._version_rows: list[list[dict[str, Any]]] = list(version_rows)
        self._rules = rules
        self.executed: list[str] = []
        self._last_query = ""

    def execute(self, query: str, ignore_error: bool = False) -> None:
        self._last_query = "version" if VERSION_QUERY_MARKER in query else "rules"
        self.executed.append(self._last_query)

    def fetchall(self) -> list[dict[str, Any]]:
        if self._last_query != "version":
            return self._rules

        return self._version_rows.pop(0)


def version_row(version: int = 1, checksum: str = "0xAAAA") -> dict[str, Any]:
    return {"name": "mysql_query_rules", "version": version, "epoch": 1000, "checksum": checksum}


def make_dolphie(connection: FakeConnection, *, at_least_3: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        main_db_connection=connection,
        proxysql_query_rule_tracker=QueryRuleTracker(),
        proxysql_rule_sample=None,
        proxysql_mysql_query_rules=[],
        is_mysql_version_at_least=lambda target: at_least_3,
    )


def fetch_rules(dolphie: SimpleNamespace) -> Any:
    WorkerDataProcessor(MagicMock())._fetch_proxysql_query_rules(cast(Any, dolphie))  # pyright: ignore[reportPrivateUsage]
    return dolphie.proxysql_rule_sample


def test_a_confirmed_sample_brackets_the_rules_with_the_checksum() -> None:
    rules = [{"rule_id": "1", "hits": 3, "hits_s": 3, "active": 1}]
    # Other configuration modules share the table and must not be picked up
    checksum_response = [
        {"name": "mysql_servers", "version": 9, "epoch": 1, "checksum": "0x0000"},
        version_row(),
    ]
    connection = FakeConnection([checksum_response, checksum_response], rules)
    dolphie = make_dolphie(connection)

    sample = fetch_rules(dolphie)

    assert connection.executed == ["version", "rules", "version"]
    assert sample is not None
    assert sample.confirmed is True
    assert sample.breakpoint == BREAKPOINT_FIRST
    assert sample.identity is not None
    assert (sample.identity.runtime_version, sample.identity.checksum) == (1, "0xAAAA")
    assert dolphie.proxysql_mysql_query_rules == rules


def test_a_reload_between_the_bracketing_reads_makes_the_sample_unconfirmable() -> None:
    rules = [{"rule_id": "1", "hits": 3}]
    connection = FakeConnection([[version_row(1)], [version_row(1)]], rules)
    dolphie = make_dolphie(connection)
    first = fetch_rules(dolphie)
    assert first.confirmed is True

    # The checksum moves while this sample is being read: before/after no longer agree
    connection._version_rows = [[version_row(1)], [version_row(2, "0xBBBB")]]
    sample = fetch_rules(dolphie)

    assert sample.confirmed is False
    assert sample.identity is None
    assert sample.version_seq is None
    assert sample.breakpoint == BREAKPOINT_UNCONFIRMED
    assert sample.deltas == {"1": None}
    # The observed rules are still rendered, just not linked to a version
    assert dolphie.proxysql_mysql_query_rules == rules


def test_a_missing_checksum_row_marks_the_sample_unconfirmed() -> None:
    connection = FakeConnection([[], []], [{"rule_id": "1", "hits": 1}])
    sample = fetch_rules(make_dolphie(connection))

    assert sample is not None
    assert sample.confirmed is False
    assert sample.identity is None
    assert sample.deltas == {"1": None}


def test_zero_rules_still_confirm_a_runtime_version() -> None:
    connection = FakeConnection([[version_row()], [version_row()]], [])
    sample = fetch_rules(make_dolphie(connection))

    assert sample is not None
    assert sample.confirmed is True
    assert sample.identity is not None
    assert sample.rules == []


def test_proxysql_below_3_keeps_the_legacy_rule_id_fetch_without_a_sample() -> None:
    rules = [{"rule_id": "1", "hits": 3}]
    connection = FakeConnection([[version_row()], [version_row()]], rules)
    dolphie = make_dolphie(connection, at_least_3=False)

    fetch_rules(dolphie)

    assert connection.executed == ["rules"]
    assert dolphie.proxysql_rule_sample is None
    assert dolphie.proxysql_mysql_query_rules == rules
