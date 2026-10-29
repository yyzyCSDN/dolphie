from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import orjson
import pytest
import zstandard as zstd

from dolphie.DataTypes import ConnectionSource
from dolphie.Dolphie import Dolphie
from dolphie.Modules.GtidReplaySync import BarrierStatus, locate_gtid_barrier
from dolphie.Modules.ReplayManager import ReplayManager

UUID_A = "3e11fa47-71ca-11e1-9e33-c80aa9429562"
UUID_B = "aaaaaaaa-71ca-11e1-9e33-c80aa9429562"
TARGET_HOST_UUID = "11111111-2222-3333-4444-555555555555"

SCHEMA = """
CREATE TABLE replay_data (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME, data BLOB);
CREATE INDEX idx_replay_data_timestamp ON replay_data (timestamp);
CREATE TABLE metadata (schema_version INTEGER DEFAULT 1, host VARCHAR(255), port INTEGER, host_distro VARCHAR(255),
    connection_source VARCHAR(255), dolphie_version VARCHAR(255), compression_dict BLOB);
CREATE TABLE variable_changes (id INTEGER PRIMARY KEY AUTOINCREMENT, replay_id INTEGER, timestamp DATETIME,
    variable_name VARCHAR(255), old_value VARCHAR(255), new_value VARCHAR(255));
"""
STAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
BASE_TIME = datetime(2026, 9, 1, 12, 0, 0, tzinfo=timezone.utc)


@dataclass
class FrameSpec:
    """One recorded frame.

    gtid=None -> the gtid_executed key is absent (GTID off/not recorded). gtid="" -> present but empty.
    variables=False -> global_variables itself is missing. corrupt -> unreadable payload.
    """

    gtid: str | None = None
    server_uuid: str | None = TARGET_HOST_UUID
    corrupt: bool = False
    variables: bool = True
    stamp: datetime | None = None
    channel_executed: str | None = None


def frame_payload(spec: FrameSpec) -> dict[str, Any]:
    if not spec.variables:
        return {"global_status": {"Uptime": 1}}

    variables: dict[str, Any] = {"version": "8.4.7"}
    if spec.server_uuid is not None:
        variables["server_uuid"] = spec.server_uuid
    if spec.gtid is not None:
        variables["gtid_executed"] = spec.gtid
    payload: dict[str, Any] = {"global_variables": variables}
    if spec.channel_executed is not None:
        payload["replication_status"] = [{"Channel_Name": "", "Executed_Gtid_Set": spec.channel_executed}]
    return payload


def write_recording(
    path: Path,
    specs: list[FrameSpec],
    *,
    host: str = "host",
    port: int = 3306,
    spacing: timedelta = timedelta(seconds=2),
) -> None:
    path.unlink(missing_ok=True)
    compressor = zstd.ZstdCompressor(level=ReplayManager.COMPRESSION_LEVEL)
    connection = sqlite3.connect(path)
    try:
        connection.executescript(SCHEMA)
        connection.execute(
            "INSERT INTO metadata VALUES (?, ?, ?, ?, ?, ?, ?)",
            (2, host, port, "MySQL", ConnectionSource.mysql, "6.16.0", None),
        )
        for index, spec in enumerate(specs):
            stamp = spec.stamp or (BASE_TIME + spacing * index)
            blob = b"corrupt-payload" if spec.corrupt else compressor.compress(orjson.dumps(frame_payload(spec)))
            connection.execute(
                "INSERT INTO replay_data (timestamp, data) VALUES (?, ?)",
                (stamp.strftime(STAMP_FORMAT), blob),
            )
        connection.commit()
    finally:
        connection.close()


def make_dolphie(replay_file: Path, *, host: str = "host", port: int = 3306) -> Dolphie:
    app = SimpleNamespace(notify=lambda *args, **kwargs: None)
    dolphie = SimpleNamespace(
        host=host,
        port=port,
        host_distro="MySQL",
        connection_source=ConnectionSource.mysql,
        replay_file=str(replay_file),
        replay_dir=None,
        daemon_mode=False,
        record_for_replay=False,
        replay_retention_hours=48,
        replay_summary=False,
        app_version="6.16.0",
        app=app,
    )
    return cast(Dolphie, dolphie)


def open_manager(path: Path, *, host: str = "host") -> ReplayManager:
    manager = ReplayManager(make_dolphie(path, host=host))
    assert manager.verify_replay_file()
    return manager


def master_recording(path: Path, gtid: str, *, stamp: datetime = BASE_TIME) -> tuple[Path, int]:
    """A one-frame master recording; returns its path and the frame id to use as the barrier."""
    write_recording(path, [FrameSpec(gtid=gtid, server_uuid=UUID_A, stamp=stamp)], host="master")
    return path, 1


@pytest.fixture
def files(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "master.db", tmp_path / "replica.db"


def test_matches_the_first_target_frame_whose_server_set_contains_the_barrier(files: tuple[Path, Path]) -> None:
    master_path, replica_path = files
    master_recording(master_path, f"{UUID_A}:1-100")
    write_recording(
        replica_path,
        [
            FrameSpec(gtid=f"{UUID_A}:1-50"),
            FrameSpec(gtid=f"{UUID_A}:1-100"),
            FrameSpec(gtid=f"{UUID_A}:1-120"),
        ],
        host="replica",
    )

    master = open_manager(master_path, host="master")
    replica = open_manager(replica_path, host="replica")
    try:
        cursor_before = replica.current_replay_id
        result = locate_gtid_barrier(master, 1, replica, 1, 3)

        assert result.status is BarrierStatus.matched
        assert result.target_replay_id == 2
        assert result.frames_scanned == 2
        assert result.master_replay_id == 1
        assert result.master_gtid_executed == f"{UUID_A}:1-100"
        assert result.target_gtid_executed == f"{UUID_A}:1-100"
        # The scan never moves either playback cursor
        assert replica.current_replay_id == cursor_before
        assert master.current_replay_id == 0
    finally:
        master.close()
        replica.close()


def test_matching_uses_gtid_sets_not_wall_clock_under_clock_skew(files: tuple[Path, Path]) -> None:
    master_path, replica_path = files
    master_recording(master_path, f"{UUID_A}:1-100", stamp=BASE_TIME)
    write_recording(
        replica_path,
        [
            # Replica's clock is an hour and seven seconds ahead; frame order still proves the barrier
            FrameSpec(gtid=f"{UUID_A}:1-100", stamp=BASE_TIME + timedelta(seconds=3607)),
        ],
        host="replica",
    )

    master = open_manager(master_path, host="master")
    replica = open_manager(replica_path, host="replica")
    try:
        result = locate_gtid_barrier(master, 1, replica, 1, 1)
        assert result.status is BarrierStatus.matched
        assert result.raw_delay_seconds == 3607.0
    finally:
        master.close()
        replica.close()


def test_a_wall_clock_close_frame_that_does_not_contain_the_barrier_is_not_a_match(
    files: tuple[Path, Path],
) -> None:
    master_path, replica_path = files
    master_recording(master_path, f"{UUID_A}:1-100")
    write_recording(
        replica_path,
        [
            # Same timestamp as the master wall clock, but the server set lags far behind
            FrameSpec(gtid=f"{UUID_A}:1-1", stamp=BASE_TIME),
            FrameSpec(gtid=f"{UUID_A}:1-100", stamp=BASE_TIME + timedelta(minutes=5)),
        ],
        host="replica",
    )

    master = open_manager(master_path, host="master")
    replica = open_manager(replica_path, host="replica")
    try:
        result = locate_gtid_barrier(master, 1, replica, 1, 2)
        assert result.status is BarrierStatus.matched
        assert result.target_replay_id == 2
    finally:
        master.close()
        replica.close()


def test_multi_uuid_barrier_and_extra_local_uuid_transactions_match(files: tuple[Path, Path]) -> None:
    master_path, replica_path = files
    master_recording(master_path, f"{UUID_A}:1-100,{UUID_B}:1-3")
    write_recording(
        replica_path,
        [
            FrameSpec(gtid=f"{UUID_A}:1-100,{UUID_B}:1-2,{TARGET_HOST_UUID}:1-9"),
            FrameSpec(gtid=f"{UUID_A}:1-100,{UUID_B}:1-3,{TARGET_HOST_UUID}:1-9"),
        ],
        host="replica",
    )

    master = open_manager(master_path, host="master")
    replica = open_manager(replica_path, host="replica")
    try:
        result = locate_gtid_barrier(master, 1, replica, 1, 2)
        assert result.status is BarrierStatus.matched
        assert result.target_replay_id == 2
    finally:
        master.close()
        replica.close()


def test_gapped_intervals_must_be_fully_covered(files: tuple[Path, Path]) -> None:
    master_path, replica_path = files
    master_recording(master_path, f"{UUID_A}:1-10:20-30")
    write_recording(
        replica_path,
        [
            FrameSpec(gtid=f"{UUID_A}:1-25"),  # covers the first interval but not 26-30
            FrameSpec(gtid=f"{UUID_A}:1-30"),  # merged coverage contains both
        ],
        host="replica",
    )

    master = open_manager(master_path, host="master")
    replica = open_manager(replica_path, host="replica")
    try:
        result = locate_gtid_barrier(master, 1, replica, 1, 2)
        assert result.status is BarrierStatus.matched
        assert result.target_replay_id == 2
        assert result.frames_scanned == 2
    finally:
        master.close()
        replica.close()


def test_not_reached_when_the_whole_monotonic_segment_lags_the_barrier(files: tuple[Path, Path]) -> None:
    master_path, replica_path = files
    master_recording(master_path, f"{UUID_A}:1-100")
    write_recording(
        replica_path,
        [FrameSpec(gtid=f"{UUID_A}:1-50"), FrameSpec(gtid=f"{UUID_A}:1-90")],
        host="replica",
    )

    master = open_manager(master_path, host="master")
    replica = open_manager(replica_path, host="replica")
    try:
        result = locate_gtid_barrier(master, 1, replica, 1, 2)
        assert result.status is BarrierStatus.not_reached
        assert result.reason is not None
        # Evidence ends at the last observed frame
        assert result.target_replay_id == 2
        assert result.target_gtid_executed == f"{UUID_A}:1-90"
        assert result.frames_scanned == 2
    finally:
        master.close()
        replica.close()


def test_a_channel_row_containing_the_barrier_is_never_used(files: tuple[Path, Path]) -> None:
    master_path, replica_path = files
    master_recording(master_path, f"{UUID_A}:1-100")
    write_recording(
        replica_path,
        [
            FrameSpec(gtid=f"{UUID_A}:1-5", channel_executed=f"{UUID_A}:1-100"),
        ],
        host="replica",
    )

    master = open_manager(master_path, host="master")
    replica = open_manager(replica_path, host="replica")
    try:
        result = locate_gtid_barrier(master, 1, replica, 1, 1)
        assert result.status is BarrierStatus.not_reached
    finally:
        master.close()
        replica.close()


def test_empty_barrier_is_indeterminate(files: tuple[Path, Path]) -> None:
    master_path, replica_path = files
    master_recording(master_path, "")
    write_recording(replica_path, [FrameSpec(gtid=f"{UUID_A}:1-100")], host="replica")

    master = open_manager(master_path, host="master")
    replica = open_manager(replica_path, host="replica")
    try:
        result = locate_gtid_barrier(master, 1, replica, 1, 1)
        assert result.status is BarrierStatus.indeterminate
        assert "empty gtid_executed" in (result.reason or "")
    finally:
        master.close()
        replica.close()


def test_gtid_disabled_and_missing_fields_are_indeterminate(files: tuple[Path, Path]) -> None:
    master_path, replica_path = files
    # Master frame recorded before GTID was enabled: key absent
    write_recording(master_path, [FrameSpec(gtid=None)], host="master")
    write_recording(replica_path, [FrameSpec(gtid=f"{UUID_A}:1-100")], host="replica")

    master = open_manager(master_path, host="master")
    replica = open_manager(replica_path, host="replica")
    try:
        result = locate_gtid_barrier(master, 1, replica, 1, 1)
        assert result.status is BarrierStatus.indeterminate
        assert "missing gtid_executed" in (result.reason or "")
    finally:
        master.close()
        replica.close()

    # server_uuid missing on the master is also insufficient evidence
    write_recording(master_path, [FrameSpec(gtid=f"{UUID_A}:1-1", server_uuid=None)], host="master")
    master = open_manager(master_path, host="master")
    try:
        result = locate_gtid_barrier(master, 1, replica, 1, 1)
        assert result.status is BarrierStatus.indeterminate
        assert "server_uuid" in (result.reason or "")
    finally:
        master.close()


def test_unparseable_and_tagged_barriers_are_indeterminate(files: tuple[Path, Path]) -> None:
    master_path, replica_path = files
    write_recording(replica_path, [FrameSpec(gtid=f"{UUID_A}:1-100")], host="replica")

    for bad_gtid in ("not-a-gtid-set", f"{UUID_A}:shard1:1-100"):
        write_recording(master_path, [FrameSpec(gtid=bad_gtid, server_uuid=UUID_A)], host="master")
        master = open_manager(master_path, host="master")
        replica = open_manager(replica_path, host="replica")
        try:
            result = locate_gtid_barrier(master, 1, replica, 1, 1)
            assert result.status is BarrierStatus.indeterminate
            assert result.reason is not None
        finally:
            master.close()
            replica.close()


def test_master_frame_corrupt_or_missing_is_indeterminate(files: tuple[Path, Path]) -> None:
    master_path, replica_path = files
    write_recording(master_path, [FrameSpec(gtid=f"{UUID_A}:1-1", corrupt=True)], host="master")
    write_recording(replica_path, [FrameSpec(gtid=f"{UUID_A}:1-1")], host="replica")

    master = open_manager(master_path, host="master")
    replica = open_manager(replica_path, host="replica")
    try:
        assert locate_gtid_barrier(master, 1, replica, 1, 1).status is BarrierStatus.indeterminate
        assert locate_gtid_barrier(master, 999, replica, 1, 1).status is BarrierStatus.indeterminate
    finally:
        master.close()
        replica.close()


def test_corrupt_target_frame_before_the_hit_stops_the_scan(files: tuple[Path, Path]) -> None:
    master_path, replica_path = files
    master_recording(master_path, f"{UUID_A}:1-100")
    write_recording(
        replica_path,
        [
            FrameSpec(gtid=f"{UUID_A}:1-50"),
            FrameSpec(corrupt=True),  # unreadable frame before containment
            FrameSpec(gtid=f"{UUID_A}:1-100"),
        ],
        host="replica",
    )

    master = open_manager(master_path, host="master")
    replica = open_manager(replica_path, host="replica")
    try:
        result = locate_gtid_barrier(master, 1, replica, 1, 3)
        assert result.status is BarrierStatus.indeterminate
        assert "corrupt" in (result.reason or "")
        assert result.frames_scanned == 1
        # No state mutation
        assert replica.current_replay_id == 0
    finally:
        master.close()
        replica.close()


def test_a_missing_row_inside_the_segment_is_indeterminate(files: tuple[Path, Path]) -> None:
    master_path, replica_path = files
    master_recording(master_path, f"{UUID_A}:1-100")
    write_recording(
        replica_path,
        [FrameSpec(gtid=f"{UUID_A}:1-50"), FrameSpec(gtid=f"{UUID_A}:1-100")],
        host="replica",
    )
    # A retention purge cannot leave a hole in the middle in production, but the scan
    # must still refuse to prove containment across an unobservable frame.
    connection = sqlite3.connect(replica_path)
    try:
        connection.execute("DELETE FROM replay_data WHERE id = 1")
        connection.commit()
    finally:
        connection.close()

    master = open_manager(master_path, host="master")
    replica = open_manager(replica_path, host="replica")
    try:
        result = locate_gtid_barrier(master, 1, replica, 1, 2)
        assert result.status is BarrierStatus.indeterminate
        assert "missing" in (result.reason or "")
    finally:
        master.close()
        replica.close()


def test_server_uuid_change_is_a_generation_break(files: tuple[Path, Path]) -> None:
    master_path, replica_path = files
    master_recording(master_path, f"{UUID_A}:1-100")
    new_host_uuid = "99999999-8888-7777-6666-555555555555"
    write_recording(
        replica_path,
        [
            FrameSpec(gtid=f"{UUID_A}:1-50", server_uuid=TARGET_HOST_UUID),
            FrameSpec(gtid=f"{UUID_A}:1-100", server_uuid=new_host_uuid),
        ],
        host="replica",
    )

    master = open_manager(master_path, host="master")
    replica = open_manager(replica_path, host="replica")
    try:
        result = locate_gtid_barrier(master, 1, replica, 1, 2)
        assert result.status is BarrierStatus.indeterminate
        assert "server_uuid changed" in (result.reason or "")
        assert result.target_server_uuid == new_host_uuid
    finally:
        master.close()
        replica.close()


def test_an_observed_set_shrink_stops_as_non_monotonic(files: tuple[Path, Path]) -> None:
    master_path, replica_path = files
    master_recording(master_path, f"{UUID_A}:1-100")
    write_recording(
        replica_path,
        [
            FrameSpec(gtid=f"{UUID_A}:1-100"),  # contains the barrier already...
            FrameSpec(gtid=f"{UUID_A}:1-90"),  # ...but the history then shrinks (reset)
        ],
        host="replica",
    )

    master = open_manager(master_path, host="master")
    replica = open_manager(replica_path, host="replica")
    try:
        # First frame matches, so this is a hit: monotonicity only guards frames BEFORE the hit
        first_hit = locate_gtid_barrier(master, 1, replica, 1, 2)
        assert first_hit.status is BarrierStatus.matched
        assert first_hit.target_replay_id == 1
    finally:
        master.close()
        replica.close()

    # With the barrier beyond both frames, the shrink is encountered first and stops the scan
    write_recording(master_path, [FrameSpec(gtid=f"{UUID_A}:1-200", server_uuid=UUID_A)], host="master")

    master = open_manager(master_path, host="master")
    replica = open_manager(replica_path, host="replica")
    try:
        result = locate_gtid_barrier(master, 1, replica, 1, 2)
        assert result.status is BarrierStatus.indeterminate
        assert "no longer contains" in (result.reason or "")
        assert result.frames_scanned == 2
    finally:
        master.close()
        replica.close()


def test_a_uuid_disappearing_from_the_set_is_non_monotonic(files: tuple[Path, Path]) -> None:
    master_path, replica_path = files
    master_recording(master_path, f"{UUID_A}:1-6,{UUID_B}:1-4")
    write_recording(
        replica_path,
        [
            FrameSpec(gtid=f"{UUID_A}:1-5,{UUID_B}:1-3"),
            FrameSpec(gtid=f"{UUID_A}:1-6"),  # UUID_B transactions vanished
            FrameSpec(gtid=f"{UUID_A}:1-6,{UUID_B}:1-4"),
        ],
        host="replica",
    )

    master = open_manager(master_path, host="master")
    replica = open_manager(replica_path, host="replica")
    try:
        result = locate_gtid_barrier(master, 1, replica, 1, 3)
        assert result.status is BarrierStatus.indeterminate
        assert "no longer contains" in (result.reason or "")
    finally:
        master.close()
        replica.close()


def test_gap_fill_between_two_monotonic_frames_is_accepted(files: tuple[Path, Path]) -> None:
    master_path, replica_path = files
    master_recording(master_path, f"{UUID_A}:1-30")
    write_recording(
        replica_path,
        [
            FrameSpec(gtid=f"{UUID_A}:1-10:20-25"),
            FrameSpec(gtid=f"{UUID_A}:1-30"),  # fills 11-19 and 26-30, contains the prior frame
        ],
        host="replica",
    )

    master = open_manager(master_path, host="master")
    replica = open_manager(replica_path, host="replica")
    try:
        result = locate_gtid_barrier(master, 1, replica, 1, 2)
        assert result.status is BarrierStatus.matched
        assert result.target_replay_id == 2
    finally:
        master.close()
        replica.close()


def test_target_frame_with_gtid_off_unparseable_or_tagged_is_indeterminate(files: tuple[Path, Path]) -> None:
    master_path, replica_path = files
    master_recording(master_path, f"{UUID_A}:1-100")

    for second_spec in (
        FrameSpec(gtid=""),
        FrameSpec(gtid="bogus"),
        FrameSpec(gtid=f"{UUID_A}:tag:1-100"),
        FrameSpec(gtid=f"{UUID_A}:1-100", server_uuid=None),
        FrameSpec(gtid=f"{UUID_A}:1-100", variables=False),
    ):
        write_recording(replica_path, [FrameSpec(gtid=f"{UUID_A}:1-50"), second_spec], host="replica")
        master = open_manager(master_path, host="master")
        replica = open_manager(replica_path, host="replica")
        try:
            result = locate_gtid_barrier(master, 1, replica, 1, 2)
            assert result.status is BarrierStatus.indeterminate, result.reason
        finally:
            master.close()
            replica.close()
        replica_path.unlink()


def test_scan_honors_the_marked_segment_and_record_order_not_timestamps(
    files: tuple[Path, Path],
) -> None:
    master_path, replica_path = files
    master_recording(master_path, f"{UUID_A}:1-100")
    # Timestamps step backwards like a clock reset; ids still define record order
    out_of_order = BASE_TIME + timedelta(minutes=10)
    write_recording(
        replica_path,
        [
            FrameSpec(gtid=f"{UUID_A}:1-100", stamp=out_of_order),
            FrameSpec(gtid=f"{UUID_A}:1-100", stamp=BASE_TIME),
            FrameSpec(gtid=f"{UUID_A}:1-100", stamp=BASE_TIME - timedelta(minutes=5)),
        ],
        host="replica",
    )

    master = open_manager(master_path, host="master")
    replica = open_manager(replica_path, host="replica")
    try:
        # Segment restricted to frames 2-3: frame 1 (also a hit) is outside the segment
        result = locate_gtid_barrier(master, 1, replica, 2, 3)
        assert result.status is BarrierStatus.matched
        assert result.target_replay_id == 2
        assert result.segment_start_id == 2
        assert result.segment_end_id == 3

        # No monotonic bridge to frames before the segment is required
        assert locate_gtid_barrier(master, 1, replica, 3, 3).target_replay_id == 3

        assert locate_gtid_barrier(master, 1, replica, 3, 2).status is BarrierStatus.indeterminate
        assert locate_gtid_barrier(master, 1, replica, 0, 2).status is BarrierStatus.indeterminate
    finally:
        master.close()
        replica.close()


def test_segment_markers_default_to_the_whole_file_and_validate_order(files: tuple[Path, Path]) -> None:
    master_path, replica_path = files
    master_recording(master_path, f"{UUID_A}:1-1")
    write_recording(replica_path, [FrameSpec(gtid=f"{UUID_A}:1-{n}") for n in range(1, 5)], host="replica")

    replica = open_manager(replica_path, host="replica")
    try:
        assert replica.gtid_segment_window() == (1, 4)

        replica.current_replay_id = 2
        replica.current_replay_timestamp = replica.frame_timestamp(2)
        assert replica.mark_gtid_segment_start() == (
            2,
            (BASE_TIME + timedelta(seconds=2)).strftime(STAMP_FORMAT),
        )
        assert replica.mark_gtid_segment_end(3) is not None
        assert replica.gtid_segment_window() == (2, 3)

        # Existing frames outside the marked order are rejected; missing ids are too
        with pytest.raises(ValueError, match="first frame"):
            replica.mark_gtid_segment_start(4)
        with pytest.raises(ValueError, match="last frame"):
            replica.mark_gtid_segment_end(1)
        with pytest.raises(ValueError, match="does not exist"):
            replica.mark_gtid_segment_start(999)
    finally:
        replica.close()


def test_purged_markers_invalidate_the_segment_window(files: tuple[Path, Path]) -> None:
    master_path, replica_path = files
    master_recording(master_path, f"{UUID_A}:1-1")
    write_recording(replica_path, [FrameSpec(gtid=f"{UUID_A}:1-1") for _ in range(3)], host="replica")

    replica = open_manager(replica_path, host="replica")
    connection = sqlite3.connect(replica_path)
    try:
        replica.mark_gtid_segment_start(1)
        replica.mark_gtid_segment_end(3)
        connection.execute("DELETE FROM replay_data WHERE id = 1")
        connection.commit()
    finally:
        connection.close()

    try:
        with pytest.raises(ValueError, match="marked first frame"):
            replica.gtid_segment_window()
    finally:
        replica.close()


def test_cursor_positioning_methods_target_the_matched_frames(files: tuple[Path, Path]) -> None:
    master_path, replica_path = files
    master_recording(master_path, f"{UUID_A}:1-1")
    write_recording(replica_path, [FrameSpec(gtid=f"{UUID_A}:1-1") for _ in range(3)], host="replica")

    replica = open_manager(replica_path, host="replica")
    try:
        # Active side: armed one before the frame for the next worker tick
        assert replica.arm_frame_for_playback(3)
        assert replica.current_replay_id == 2
        assert replica.current_replay_timestamp == (BASE_TIME + timedelta(seconds=4)).strftime(STAMP_FORMAT)

        # Inactive side: left on the frame so the tab switch reloads it
        assert replica.position_at_frame(2)
        assert replica.current_replay_id == 2
        assert replica.current_replay_timestamp == (BASE_TIME + timedelta(seconds=2)).strftime(STAMP_FORMAT)

        assert replica.arm_frame_for_playback(999) is False
    finally:
        replica.close()
