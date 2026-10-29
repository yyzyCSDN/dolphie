"""End-to-end playback test for the GTID barrier comparison across two replay tabs."""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path
from typing import Any

import orjson
import zstandard as zstd
from textual.widgets import Button, Static

from dolphie.DataTypes import ConnectionStatus
from dolphie.Modules.ReplayManager import ReplayManager
from dolphie.Modules.TabManager import Tab
from dolphie.Widgets.GtidCompareModal import GtidCompareModal
from dolphie.Widgets.GtidCompareResultScreen import GtidCompareResultScreen
from tests.dolphie.replay_files import read_compression_dict, read_replay_rows
from tests.integration.harness import REPLAYS, DolphieHarness, HarnessApp, replay_config

UUID_A = "3e11fa47-71ca-11e1-9e33-c80aa9429562"
REPLICA_UUID = "11111111-2222-3333-4444-555555555555"


def rewrite_gtid_frames(path: Path, gtid_by_id: dict[int, str], server_uuid: str) -> None:
    """Overwrites server-level gtid_executed/server_uuid in every row of a copied recording."""
    dictionary = read_compression_dict(path)
    compressor = zstd.ZstdCompressor(dict_data=zstd.ZstdCompressionDict(dictionary) if dictionary else None)
    rows: list[tuple[int, dict[str, Any]]] = read_replay_rows(path, "id, data")

    connection = sqlite3.connect(path)
    try:
        for replay_id, data in rows:
            variables = data["global_variables"]
            variables["server_uuid"] = server_uuid
            variables["gtid_executed"] = gtid_by_id[replay_id]
            connection.execute(
                "UPDATE replay_data SET data = ? WHERE id = ?",
                (compressor.compress(orjson.dumps(data)), replay_id),
            )
        connection.commit()
    finally:
        connection.close()


def prepare_recordings(tmp_path: Path) -> tuple[Path, Path]:
    master = tmp_path / "master.db"
    replica = tmp_path / "replica.db"
    shutil.copy(REPLAYS / "mysql.db", master)
    shutil.copy(REPLAYS / "mysql.db", replica)

    # Master holds the barrier at frame 1 already; the replica catches up half a segment later
    rewrite_gtid_frames(master, dict.fromkeys(range(1, 9), f"{UUID_A}:1-100"), UUID_A)
    replica_sets = {
        **dict.fromkeys(range(1, 5), f"{UUID_A}:1-50"),
        **dict.fromkeys(range(5, 9), f"{UUID_A}:1-100"),
    }
    rewrite_gtid_frames(replica, replica_sets, REPLICA_UUID)
    return master, replica


async def add_replay_tab(app: HarnessApp, harness: DolphieHarness, path: Path, name: str) -> Tab:
    """Adds a second tab playing a replay file, mirroring setup_host_tab's replay branch.

    The tab is created active (the production flow) so its first frame loads before any
    switch-back can render it with empty state.
    """
    tab = await app.tab_manager.create_tab(tab_name=name)
    tab.dolphie.replay_file = str(path)
    tab.replay_manager = ReplayManager(tab.dolphie)
    assert tab.replay_manager.verify_replay_file()
    app.tab_manager.update_connection_status(tab, ConnectionStatus.connected)
    app.run_worker_replay(tab.id)
    manager = tab.replay_manager
    await harness.wait_for(lambda: manager.current_replay_id >= 1, message=f"{name} first frame")
    return tab


async def activate_tab(harness: DolphieHarness, app: HarnessApp, tab: Tab, frame: int = 1) -> None:
    manager = tab.replay_manager
    assert manager is not None
    app.tab_manager.host_tabs.active = tab.id
    await harness.wait_for(
        lambda: app.tab_manager.active_tab is tab and manager.current_replay_id >= frame,
        message=f"tab {tab.name} frame {frame}",
    )


async def step_to_frame(harness: DolphieHarness, app: HarnessApp, frame: int) -> None:
    manager = harness.tab.replay_manager
    assert manager is not None
    while manager.current_replay_id < frame:
        # A step requested while a worker is still rendering is dropped, so idle first
        await harness.wait_for_worker_idle()
        await _step_one(harness, app, manager)
    await harness.wait_for_worker_idle()


async def _step_one(harness: DolphieHarness, app: HarnessApp, manager: ReplayManager) -> None:
    before = manager.current_replay_id
    app.action_replay_forward()
    await harness.wait_for(lambda: manager.current_replay_id > before, message=f"frame > {before}")


async def click_screen_button(harness: DolphieHarness, app: HarnessApp, selector: str) -> None:
    """Presses a button on the pushed modal/result screen (app.query_one stays on the base screen)."""
    app.screen.query_one(selector, Button).press()
    await harness._settle()


async def wait_for_result_screen(harness: DolphieHarness, app: HarnessApp) -> GtidCompareResultScreen:
    def ready() -> bool:
        screen = app.screen
        return isinstance(screen, GtidCompareResultScreen) and bool(screen.query("#gtid_result_body"))

    await harness.wait_for(ready, message="comparison result mounted")
    assert isinstance(app.screen, GtidCompareResultScreen)
    return app.screen


async def test_gtid_barrier_match_moves_both_cursors_and_jumps_to_the_target_graph(
    tmp_path: Path,
) -> None:
    master_path, replica_path = prepare_recordings(tmp_path)
    app = HarnessApp(replay_config(master_path, refresh_interval=3600))
    async with app.run_test(size=(180, 55)) as pilot:
        harness = DolphieHarness(app, pilot)
        await harness.wait_for_replay_frame()
        await harness.click_button("#pause_button")
        await harness.wait_for_worker_idle()

        master_tab = harness.tab
        master_manager = master_tab.replay_manager
        assert master_manager is not None
        assert master_manager.current_replay_id == 1

        replica_tab = await add_replay_tab(app, harness, replica_path, "replica")
        replica_manager = replica_tab.replay_manager
        assert replica_manager is not None

        # Mark the target segment #2 -> #7 on the replica tab
        await activate_tab(harness, app, replica_tab)
        await harness.click_button("#pause_button")
        await harness.wait_for_worker_idle()

        await step_to_frame(harness, app, 2)
        app.action_gtid_mark_segment_start()
        assert replica_manager.gtid_segment_start_id == 2
        await step_to_frame(harness, app, 7)
        app.action_gtid_mark_segment_end()
        assert replica_manager.gtid_segment_end_id == 7
        assert "GTID segment" in str(harness.tab.dashboard_replay.render())

        # Back on the master frame, run the comparison against the replica recording
        await activate_tab(harness, app, master_tab, frame=1)
        await harness.wait_for_worker_idle()
        assert master_manager.current_replay_id == 1

        app.action_gtid_barrier_compare()
        await pilot.pause(0)
        assert isinstance(app.screen, GtidCompareModal)

        await click_screen_button(harness, app, "#submit")
        screen = await wait_for_result_screen(harness, app)
        assert screen.result.matched
        assert screen.result.target_replay_id == 5
        assert screen.result.master_replay_id == 1
        body = str(screen.query_one("#gtid_result_body", Static).render())
        assert "MATCHED" in body
        assert str(screen.result.master_timestamp or "") in body
        assert harness.notifications_with("both cursors updated")

        # Both cursors moved together; failure is impossible here, positions are exact
        assert master_manager.current_replay_id == 1
        assert replica_manager.current_replay_id == 5
        assert replica_tab.dolphie.pause_refresh is True
        assert master_tab.dolphie.pause_refresh is True

        # Jump to the target side: switching tabs reloads frame #5 and its graph window
        await click_screen_button(harness, app, "#goto_target")
        await harness.wait_for(lambda: app.tab_manager.active_tab is replica_tab, message="target tab")
        await harness.wait_for_worker_idle()
        assert replica_manager.current_replay_id == 5
        assert replica_tab.dolphie.global_variables["gtid_executed"] == f"{UUID_A}:1-100"

        harness.fail_on_errors()


async def test_gtid_barrier_not_reached_leaves_cursors_in_place(tmp_path: Path) -> None:
    master_path, replica_path = prepare_recordings(tmp_path)
    app = HarnessApp(replay_config(master_path, refresh_interval=3600))
    async with app.run_test(size=(180, 55)) as pilot:
        harness = DolphieHarness(app, pilot)
        await harness.wait_for_replay_frame()
        await harness.click_button("#pause_button")
        await harness.wait_for_worker_idle()
        master_tab = harness.tab
        master_manager = master_tab.replay_manager

        replica_tab = await add_replay_tab(app, harness, replica_path, "replica")
        replica_manager = replica_tab.replay_manager
        assert replica_manager is not None

        await activate_tab(harness, app, replica_tab)
        await harness.click_button("#pause_button")
        await harness.wait_for_worker_idle()
        await step_to_frame(harness, app, 2)
        app.action_gtid_mark_segment_start()
        await step_to_frame(harness, app, 4)
        app.action_gtid_mark_segment_end()

        await activate_tab(harness, app, master_tab, frame=1)
        await harness.wait_for_worker_idle()

        app.action_gtid_barrier_compare()
        await pilot.pause(0)
        await click_screen_button(harness, app, "#submit")
        screen = await wait_for_result_screen(harness, app)
        assert not screen.result.matched
        body = screen.query_one("#gtid_result_body", Static).render()
        assert "NOT REACHED" in str(body)
        assert harness.notifications_with("not reach the barrier")
        # Positions preserved on failure
        assert replica_manager.current_replay_id == 4
        assert master_manager is not None
        assert master_manager.current_replay_id == 1

        harness.fail_on_errors()


async def test_gtid_comparison_without_a_target_replay_warns(tmp_path: Path) -> None:
    master_path, _ = prepare_recordings(tmp_path)
    app = HarnessApp(replay_config(master_path, refresh_interval=3600))
    async with app.run_test(size=(180, 55)) as pilot:
        harness = DolphieHarness(app, pilot)
        await harness.wait_for_replay_frame()

        app.action_gtid_barrier_compare()
        await pilot.pause(0)
        assert len(app.screen_stack) == 1
        assert harness.notifications_with("No GTID comparison target")

        harness.fail_on_errors()
