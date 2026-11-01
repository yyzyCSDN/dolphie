"""Query-rule version history round-trips through interactive ProxySQL recordings."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

import orjson
import pytest
import zstandard as zstd

from dolphie.DataTypes import ConnectionSource
from dolphie.Modules.ProxySQLQueryRuleTracker import QueryRuleTracker, RuleSample
from dolphie.Modules.ReplayManager import ProxySQLReplayData, ReplayManager
from tests.dolphie.Modules.test_ReplayManager import SCHEMA, make_dolphie
from tests.integration.harness import REPLAYS, DolphieHarness, HarnessApp, replay_config

FRAMES = 4


def rule(rule_id: int, hits: int, match_pattern: str) -> dict[str, Any]:
    return {
        "rule_id": rule_id,
        "hits": hits,
        "hits_s": hits,
        "active": 1,
        "apply": 1,
        "log": 0,
        "match_pattern": match_pattern,
    }


def recorded_samples() -> list[RuleSample]:
    """A recording that observes v1, then a LOAD that deletes rule 2 and changes rule 1."""
    tracker = QueryRuleTracker()
    samples = [
        tracker.record(
            [rule(1, 0, "^SELECT"), rule(2, 0, "^COMMIT")],
            runtime_version=1,
            epoch=1000,
            checksum="0xAAAA",
            connection_generation=1,
            confirmed=True,
            observed_at=0.0,
        ),
        tracker.record(
            [rule(1, 10, "^SELECT"), rule(2, 20, "^COMMIT")],
            runtime_version=1,
            epoch=1000,
            checksum="0xAAAA",
            connection_generation=1,
            confirmed=True,
            observed_at=1.0,
        ),
        tracker.record(
            [rule(1, 0, "^INSERT")],
            runtime_version=2,
            epoch=2000,
            checksum="0xBBBB",
            connection_generation=1,
            confirmed=True,
            observed_at=2.0,
        ),
        tracker.record(
            [rule(1, 8, "^INSERT")],
            runtime_version=2,
            epoch=2000,
            checksum="0xBBBB",
            connection_generation=1,
            confirmed=True,
            observed_at=4.0,
        ),
    ]
    return samples


def make_row(index: int, sample: RuleSample | None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "global_status": {
            "ProxySQL_Uptime": 100 + index,
            "MySQL_Thread_Workers": 2,
            "Client_Connections_connected": 1,
            "Active_Transactions": 0,
            "replay_polling_latency": 0.1,
        },
        "global_variables": {
            "admin-version": "3.0.11",
            "mysql-server_version": "8.4.0",
            "mysql-max_connections": 512,
        },
        "processlist": [],
        "metric_manager": {"datetimes": [f"2026-09-01 12:00:0{index}"]},
        "command_stats": [],
        "hostgroup_summary": [],
    }
    if sample is not None:
        row["query_rules"] = sample.to_replay()
    return row


def write_proxysql_replay(
    path: Path,
    samples: Sequence[RuleSample | None],
    *,
    start: datetime = datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc),
) -> None:
    compressor = zstd.ZstdCompressor(level=ReplayManager.COMPRESSION_LEVEL)
    connection = sqlite3.connect(path)
    try:
        connection.executescript(SCHEMA)
        connection.execute(
            "INSERT INTO metadata VALUES (?, ?, ?, ?, ?, ?, ?)",
            (2, "proxy1", 6032, "ProxySQL", ConnectionSource.proxysql, "6.16.0", None),
        )
        for index, sample in enumerate(samples):
            payload = orjson.dumps(make_row(index, sample))
            timestamp = (start + timedelta(seconds=index * 2)).strftime("%Y-%m-%d %H:%M:%S")
            connection.execute(
                "INSERT INTO replay_data (timestamp, data) VALUES (?, ?)",
                (timestamp, compressor.compress(payload)),
            )
        connection.commit()
    finally:
        connection.close()


def open_manager(replay_file: Path) -> ReplayManager:
    dolphie, _ = make_dolphie(replay_file, connection_source=ConnectionSource.proxysql)
    manager = ReplayManager(dolphie)
    assert manager.verify_replay_file()
    return manager


def test_playback_restores_each_frames_rule_version_and_attribution(tmp_path: Path) -> None:
    replay_file = tmp_path / "proxysql.db"
    write_proxysql_replay(replay_file, recorded_samples())
    manager = open_manager(replay_file)

    try:
        frames: list[ProxySQLReplayData] = []
        for _ in range(FRAMES):
            data = manager.get_next_refresh_interval()
            assert isinstance(data, ProxySQLReplayData)
            frames.append(data)

        samples = [RuleSample.from_replay(frame.query_rules) for frame in frames]

        first, linked, reloaded, settled = samples
        assert first is not None
        assert linked is not None
        assert reloaded is not None
        assert settled is not None

        # Frame 1: baseline of v1 with both rules
        assert first.version_seq == 1
        assert first.deltas == {"1": None, "2": None}
        assert {row["rule_id"] for row in first.rules} == {1, 2}

        # Frame 2: rates belong to v1
        assert linked.version_seq == 1
        assert linked.deltas == {"1": 10, "2": 20}

        # Frame 3: the LOAD opens a breakpoint, rule 2's evidence is absent from the new version
        assert reloaded.version_seq == 2
        assert reloaded.breakpoint == "reload"
        assert reloaded.deltas == {"1": None}
        assert [row["rule_id"] for row in reloaded.rules] == [1]
        assert reloaded.rules[0]["match_pattern"] == "^INSERT"

        # Frame 4: attribution resumes inside v2 without reaching back across the reload
        assert settled.version_seq == 2
        assert settled.breakpoint is None
        assert settled.deltas == {"1": 8}
        assert settled.rate_for("1") == 4
    finally:
        manager.close()


def test_backward_and_seek_restore_the_rule_state_of_the_target_frame(tmp_path: Path) -> None:
    replay_file = tmp_path / "proxysql.db"
    write_proxysql_replay(replay_file, recorded_samples())
    manager = open_manager(replay_file)

    def load_frame() -> RuleSample:
        data = manager.get_next_refresh_interval()
        assert isinstance(data, ProxySQLReplayData)
        sample = RuleSample.from_replay(data.query_rules)
        assert sample is not None
        return sample

    try:
        assert [load_frame().version_seq for _ in range(FRAMES)] == [1, 1, 2, 2]

        # Back one frame: the reload frame returns with its breakpoint, not a zeroed rate
        assert manager.seek_relative(-1)
        reloaded = load_frame()
        assert reloaded is not None
        assert reloaded.version_seq == 2
        assert reloaded.breakpoint == "reload"
        assert reloaded.deltas == {"1": None}

        # Jump to the first frame: v1 and both rules come back from the recording alone
        assert manager.seek_to_timestamp("2026-09-01 12:00:00")
        first = load_frame()
        assert first is not None
        assert first.version_seq == 1
        assert {row["rule_id"] for row in first.rules} == {1, 2}
        assert first.rules[0]["match_pattern"] == "^SELECT"
    finally:
        manager.close()


def test_frames_without_query_rule_data_play_back_as_unavailable(tmp_path: Path) -> None:
    # An old recording (old Dolphie, or a daemon file) simply has no query_rules key
    replay_file = tmp_path / "old-proxysql.db"
    write_proxysql_replay(replay_file, [None, None])
    manager = open_manager(replay_file)

    try:
        for _ in range(2):
            data = manager.get_next_refresh_interval()
            assert isinstance(data, ProxySQLReplayData)
            assert data.query_rules is None
            assert RuleSample.from_replay(data.query_rules) is None
    finally:
        manager.close()


async def test_tui_round_trip_renders_the_recorded_versions_and_the_reload_boundary(
    tmp_path: Path,
) -> None:
    replay_file = tmp_path / "proxysql.db"
    write_proxysql_replay(replay_file, recorded_samples())
    app = HarnessApp(replay_config(replay_file, refresh_interval=3600))

    async with app.run_test(size=(220, 60)) as pilot:
        harness = DolphieHarness(app, pilot)
        await harness.wait_for_replay_frame()
        await harness.click_button("#pause_button")
        await harness.wait_for_worker_idle()

        await harness.press("5")
        assert harness.dolphie.panels.proxysql_mysql_query_rules.visible
        table = harness.tab.proxysql_mysql_query_rules_datatable

        def status() -> str:
            return str(harness.tab.proxysql_mysql_query_rules_status.content)

        assert table.row_count == 2
        assert "Rules version" in status()
        assert "v1" in status()

        async def step(frame: int) -> None:
            await harness.press("right_square_bracket")
            await harness.wait_for(lambda: harness.replay_manager.current_replay_id == frame, message=f"frame {frame}")
            await harness.wait_for_worker_idle()

        await step(2)
        assert "v1" in status()

        await step(3)
        assert table.row_count == 1
        assert "v2" in status()
        assert "reloaded" in status()

        await step(4)
        assert table.row_count == 1
        assert "reloaded" not in status()
        labels = [str(column.label) for column in table.columns.values()]
        hits_s_index = labels.index("Hits/s")
        # 8 hits across the recorded 2-second v2 interval = 4/s, computed from the frame alone
        assert cast(str, table.get_row("1")[hits_s_index]).endswith("4")

    assert harness.error_notifications == []


async def test_tui_playback_of_an_old_recording_marks_rule_history_unavailable(tmp_path: Path) -> None:
    replay_file = tmp_path / "old-proxysql.db"
    write_proxysql_replay(replay_file, [None, None])
    app = HarnessApp(replay_config(replay_file, refresh_interval=3600))

    async with app.run_test(size=(220, 60)) as pilot:
        harness = DolphieHarness(app, pilot)
        await harness.wait_for_replay_frame()
        await harness.press("5")

        assert harness.dolphie.panels.proxysql_mysql_query_rules.visible
        status = str(harness.tab.proxysql_mysql_query_rules_status.content)
        assert "unavailable" in status
        assert harness.tab.proxysql_mysql_query_rules_datatable.row_count == 0

    assert harness.error_notifications == []


@pytest.mark.parametrize("replay_file", [REPLAYS / "proxysql.db"])
def test_the_committed_proxysql_recording_has_no_rule_history(replay_file: Path) -> None:
    manager = open_manager(replay_file)
    try:
        data = manager.get_next_refresh_interval()
        assert isinstance(data, ProxySQLReplayData)
        assert RuleSample.from_replay(data.query_rules) is None
    finally:
        manager.close()
