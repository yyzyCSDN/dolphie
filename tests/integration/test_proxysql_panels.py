"""ProxySQL detection and every ProxySQL panel, rendered against a live ProxySQL admin interface."""

from __future__ import annotations

import asyncio
from pathlib import Path

from dolphie.DataTypes import ConnectionSource
from dolphie.Modules.ProxySQLQueryRuleTracker import BREAKPOINT_RELOAD
from tests.dolphie.replay_files import read_replay_rows
from tests.integration.harness import (
    frontend_traffic,
    make_config,
    playback_config,
    query,
    run_dolphie,
)
from tests.integration.servers import Server


async def test_detects_proxysql_from_the_admin_port(proxysql_server: Server, tmp_path: Path) -> None:
    async with run_dolphie(make_config(proxysql_server, tmp_path)) as harness:
        await harness.wait_for_polls(2)
        dolphie = harness.dolphie

        assert dolphie.connection_source == ConnectionSource.proxysql
        assert dolphie.host_distro == ConnectionSource.proxysql
        assert dolphie.host_version is not None
        assert dolphie.host_version.startswith(proxysql_server.version)
        assert "Queries" in dolphie.global_status
        with frontend_traffic():
            await harness.wait_for(lambda: bool(dolphie.proxysql_command_stats), message="command stats")


async def test_runs_with_every_panel_open(proxysql_server: Server, tmp_path: Path) -> None:
    """The smoke test: five seconds of polling under traffic with every panel showing."""
    with frontend_traffic():
        async with run_dolphie(make_config(proxysql_server, tmp_path)) as harness:
            await harness.wait_for_polls(1)
            await harness.press("3", "4", "5", "6")
            for panel in (
                "graphs",
                "proxysql_hostgroup_summary",
                "proxysql_mysql_query_rules",
                "proxysql_command_stats",
            ):
                assert getattr(harness.dolphie.panels, panel).visible, panel

            # The MySQL-only panel keys are refused
            await harness.press("7", "8")
            assert not harness.dolphie.panels.pfs_metrics.visible
            assert not harness.dolphie.panels.statements_summary.visible

            assert await harness.run_for(5) >= 5


async def test_hostgroup_summary_panel_shows_the_backend(proxysql_server: Server, tmp_path: Path) -> None:
    async with run_dolphie(make_config(proxysql_server, tmp_path)) as harness:
        await harness.wait_for_polls(1)
        await harness.press("4")
        assert harness.dolphie.panels.proxysql_hostgroup_summary.visible

        with frontend_traffic():
            await harness.wait_for(
                lambda: any(row.get("srv_host") == "mysql84" for row in harness.dolphie.proxysql_hostgroup_summary),
                message="mysql84 backend in hostgroup summary",
            )
            await harness.next_poll()
            assert harness.tab.proxysql_hostgroup_summary_datatable.row_count >= 1

            await harness.wait_for(
                lambda: harness.dolphie.metric_manager.metrics.dml.Queries.latest_value() not in (None, 0),
                message="frontend traffic in the Queries metric",
            )


async def test_query_rules_panel_lists_configured_rules(proxysql_server: Server, tmp_path: Path) -> None:
    async with run_dolphie(make_config(proxysql_server, tmp_path)) as harness:
        await harness.wait_for_polls(1)
        await harness.press("5")
        assert harness.dolphie.panels.proxysql_mysql_query_rules.visible

        with frontend_traffic():
            await harness.wait_for(lambda: len(harness.dolphie.proxysql_mysql_query_rules) >= 2, message="query rules")
            await harness.next_poll()
            assert harness.tab.proxysql_mysql_query_rules_datatable.row_count >= 2


async def test_command_stats_panel(proxysql_server: Server, tmp_path: Path) -> None:
    async with run_dolphie(make_config(proxysql_server, tmp_path)) as harness:
        await harness.wait_for_polls(1)
        await harness.press("6")
        assert harness.dolphie.panels.proxysql_command_stats.visible

        with frontend_traffic():
            await harness.wait_for_polls(3)
            assert harness.tab.proxysql_command_stats_datatable.row_count >= 1
            select_rows = [r for r in harness.dolphie.proxysql_command_stats if r.get("Command") == "SELECT"]
            assert select_rows
            assert int(select_rows[0]["Total_cnt"]) > 0


def admin(server: Server, sql: str) -> None:
    """Run an admin statement (including LOAD) on its own connection, off the event loop."""
    query(server, sql)


async def test_query_rule_reloads_are_versioned_with_a_breakpoint_and_keep_old_evidence(
    proxysql_server: Server, tmp_path: Path
) -> None:
    """A mid-sample reload changes the effective version; identical re-LOADs stay distinct."""
    async with run_dolphie(make_config(proxysql_server, tmp_path)) as harness:
        await harness.wait_for_polls(1)
        await harness.press("5")

        def confirmed_sample():
            sample = harness.dolphie.proxysql_rule_sample
            return sample if sample is not None and sample.confirmed else None

        with frontend_traffic():
            await harness.wait_for(
                lambda: (s := confirmed_sample()) is not None and s.version_seq == 1,
                message="the first confirmed rule version",
            )
            first_sample = confirmed_sample()
            assert first_sample is not None
            assert first_sample.identity is not None
            start_version = first_sample.identity.runtime_version

            def status() -> str:
                return str(harness.tab.proxysql_mysql_query_rules_status.content)

            assert "Rules version" in status()

            try:
                # rule_id 2 gets a different body and is loaded to runtime
                await asyncio.to_thread(
                    admin,
                    proxysql_server,
                    "UPDATE mysql_query_rules SET match_pattern='^SELECT 2' WHERE rule_id=2",
                )
                await asyncio.to_thread(admin, proxysql_server, "LOAD MYSQL QUERY RULES TO RUNTIME")

                # The poll after the LOAD is a breakpoint: no rate crosses the version, and
                # rule_id 2's body is the new text
                await harness.wait_for(
                    lambda: (
                        (s := harness.dolphie.proxysql_rule_sample) is not None and s.breakpoint == BREAKPOINT_RELOAD
                    ),
                    message="the reload breakpoint",
                )
                reloaded = harness.dolphie.proxysql_rule_sample
                assert reloaded is not None
                assert reloaded.identity is not None
                assert reloaded.identity.runtime_version > start_version
                assert reloaded.deltas
                assert all(delta is None for delta in reloaded.deltas.values())
                assert "reloaded" in status()
                rule_two = next(row for row in reloaded.rules if str(row["rule_id"]) == "2")
                assert rule_two["match_pattern"] == "^SELECT 2"

                # Loading the same body once more is still a new version, never merged back
                checksum_after_reload = reloaded.identity.checksum
                reloaded_runtime_version = reloaded.identity.runtime_version
                await asyncio.to_thread(admin, proxysql_server, "LOAD MYSQL QUERY RULES TO RUNTIME")
                await harness.wait_for(
                    lambda: (
                        (s := harness.dolphie.proxysql_rule_sample) is not None
                        and s.identity is not None
                        and s.identity.runtime_version > reloaded_runtime_version
                        and s.breakpoint == BREAKPOINT_RELOAD
                    ),
                    message="the no-op reload breakpoint",
                )
                same_body_reload = harness.dolphie.proxysql_rule_sample
                assert same_body_reload is not None
                assert same_body_reload.identity is not None
                assert same_body_reload.version_seq is not None
                assert reloaded.version_seq is not None
                assert same_body_reload.identity.checksum == checksum_after_reload
                assert same_body_reload.version_seq == reloaded.version_seq + 1

                # Rates then resume inside the new version
                await harness.wait_for(
                    lambda: (
                        (s := harness.dolphie.proxysql_rule_sample) is not None
                        and s.breakpoint is None
                        and any(delta is not None for delta in s.deltas.values())
                    ),
                    message="hits/s attributed inside the reloaded version",
                )
            finally:
                await asyncio.to_thread(
                    admin,
                    proxysql_server,
                    "UPDATE mysql_query_rules SET match_pattern='^SELECT' WHERE rule_id=2",
                )
                await asyncio.to_thread(admin, proxysql_server, "LOAD MYSQL QUERY RULES TO RUNTIME")

            # The first version's evidence (rule 2 with the original body) is retained even
            # though rule_id 2 now carries different text
            evidence_rows = [
                row for rows in harness.dolphie.proxysql_query_rule_tracker.evidence.values() for row in rows
            ]
            assert any(str(row["rule_id"]) == "2" and row["match_pattern"] == "^SELECT" for row in evidence_rows)


async def test_query_rule_version_history_records_and_replays(proxysql_server: Server, tmp_path: Path) -> None:
    """A recording with the panel open carries the rule versions, and playback restores them."""
    async with run_dolphie(make_config(proxysql_server, tmp_path, record_for_replay=True)) as harness:
        await harness.wait_for_polls(2)
        await harness.press("5")
        with frontend_traffic():
            await harness.wait_for(
                lambda: (s := harness.dolphie.proxysql_rule_sample) is not None and s.confirmed,
                message="rules captured while recording",
            )
            first_capture = harness.dolphie.proxysql_rule_sample
            assert first_capture is not None
            assert first_capture.identity is not None
            first_runtime_version = first_capture.identity.runtime_version

            await asyncio.to_thread(
                admin,
                proxysql_server,
                "UPDATE mysql_query_rules SET match_pattern='^SELECT 9' WHERE rule_id=2",
            )
            await asyncio.to_thread(admin, proxysql_server, "LOAD MYSQL QUERY RULES TO RUNTIME")
            await harness.wait_for(
                lambda: (
                    (s := harness.dolphie.proxysql_rule_sample) is not None
                    and s.identity is not None
                    and s.identity.runtime_version > first_runtime_version
                ),
                message="the reload captured in the recording",
            )
            await harness.wait_for_polls(2)

        replay_file = Path(harness.replay_manager.replay_file)

    try:
        rows = [data for (data,) in read_replay_rows(replay_file, "data") if "query_rules" in data]
        assert rows, "the recording captured no query-rule samples"
        runtime_versions = {data["query_rules"]["identity"]["runtime_version"] for data in rows}
        assert len(runtime_versions) >= 2
        # A reload frame carries the boundary; hits/s was not computed across it
        reload_frames = [data for data in rows if data["query_rules"]["breakpoint"] == "reload"]
        assert reload_frames
        assert all(all(delta is None for delta in data["query_rules"]["deltas"].values()) for data in reload_frames)
        # Rule 2's two bodies are both present in the recording, keyed by different versions
        bodies = {
            (
                data["query_rules"]["identity"]["runtime_version"],
                next(
                    (row["match_pattern"] for row in data["query_rules"]["rules"] if str(row["rule_id"]) == "2"),
                    None,
                ),
            )
            for data in rows
        }
        assert (first_runtime_version, "^SELECT") in bodies
        assert any(pattern == "^SELECT 9" for _, pattern in bodies)

        async with run_dolphie(playback_config(proxysql_server, tmp_path, replay_file)) as playback:
            await playback.wait_for_replay_frame()
            await playback.press("5")
            status = str(playback.tab.proxysql_mysql_query_rules_status.content)
            assert "Rules version" in status
            assert "unavailable" not in status
            assert playback.tab.proxysql_mysql_query_rules_datatable.row_count >= 1
            # Playback never connects to the live proxy
            assert not playback.dolphie.main_db_connection.is_connected()
    finally:
        query(proxysql_server, "UPDATE mysql_query_rules SET match_pattern='^SELECT' WHERE rule_id=2")
        query(proxysql_server, "LOAD MYSQL QUERY RULES TO RUNTIME")


async def test_processlist_shows_frontend_sessions(proxysql_server: Server, tmp_path: Path) -> None:
    async with run_dolphie(make_config(proxysql_server, tmp_path)) as harness:
        await harness.wait_for_polls(1)
        await harness.press("i")
        assert harness.dolphie.show_idle_threads

        with frontend_traffic():
            await harness.wait_for(lambda: len(harness.dolphie.processlist_threads) >= 1, message="frontend session")
            await harness.next_poll()
            assert harness.tab.processlist_datatable.row_count >= 1


async def test_display_commands_open_a_screen(proxysql_server: Server, tmp_path: Path) -> None:
    async with run_dolphie(make_config(proxysql_server, tmp_path)) as harness:
        await harness.wait_for_polls(2)
        with frontend_traffic():
            for key in ("u", "e"):
                await harness.open_command_screen(key)
                await harness.press("escape")
                assert len(harness.app.screen_stack) == 1, key
