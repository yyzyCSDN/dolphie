"""GTID barrier scan that aligns a master replay frame with a replica replay segment.

When troubleshooting replication lag against two local recordings, the complete
server-level ``gtid_executed`` of a frame chosen on the master recording is the
barrier. The target replica recording is walked in record order (replay id order)
through the user-marked first/last segment, and the FIRST frame whose own
server-level executed set contains the barrier is the alignment point.

Deliberately, this proves one thing only: an observed set containment at two recorded
instants. It never matches by wall-clock proximity, never uses
``Retrieved_Gtid_Set`` or a per-channel ``SHOW REPLICA STATUS`` row, never infers row
data equality, and never assumes a reset that fell between two samples did not happen.
Any break in the evidence chain before the hit stops the scan as indeterminate; the
scan never resumes past the break.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from dolphie.Modules.GtidSet import GtidParseError, GtidSet, GtidTaggedError, parse_gtid_set

if TYPE_CHECKING:
    from dolphie.Modules.ReplayManager import ReplayManager


class BarrierStatus(Enum):
    """Outcome of a barrier scan."""

    matched = "matched"
    # Every frame in the segment was readable, enabled, and monotonic, but none
    # contained the barrier yet.
    not_reached = "not_reached"
    # The evidence chain is broken or insufficient; reason explains why.
    indeterminate = "indeterminate"


@dataclass(frozen=True)
class FrameObservation:
    """The server-level GTID state observed in one recorded frame."""

    replay_id: int
    timestamp: str
    server_uuid: str
    gtid_executed: str
    gtid_set: GtidSet


@dataclass
class BarrierMatchResult:
    """What the scan found. Frame fields are populated where evidence exists."""

    status: BarrierStatus
    reason: str | None = None

    master_replay_id: int | None = None
    master_timestamp: str | None = None
    master_server_uuid: str | None = None
    master_gtid_executed: str | None = None
    master_set: GtidSet | None = None

    segment_start_id: int | None = None
    segment_end_id: int | None = None

    target_replay_id: int | None = None
    target_timestamp: str | None = None
    target_server_uuid: str | None = None
    target_gtid_executed: str | None = None
    target_set: GtidSet | None = None

    # Frames walked and validated before the scan stopped (matched frame included).
    frames_scanned: int = 0
    # Raw target-minus-master timestamp difference in seconds on a match. Wall clocks
    # of two hosts are not synchronized here, so this may contain clock skew.
    raw_delay_seconds: float | None = None

    @property
    def matched(self) -> bool:
        return self.status is BarrierStatus.matched


def _build_result(
    status: BarrierStatus,
    reason: str | None,
    *,
    master: FrameObservation,
    barrier: GtidSet,
    segment_start_id: int,
    segment_end_id: int,
    frames_scanned: int,
    target: FrameObservation | None = None,
    raw_delay_seconds: float | None = None,
) -> BarrierMatchResult:
    """Builds a result with consistent, typed master and target evidence."""
    return BarrierMatchResult(
        status=status,
        reason=reason,
        master_replay_id=master.replay_id,
        master_timestamp=master.timestamp,
        master_server_uuid=master.server_uuid,
        master_gtid_executed=master.gtid_executed,
        master_set=barrier,
        segment_start_id=segment_start_id,
        segment_end_id=segment_end_id,
        target_replay_id=target.replay_id if target is not None else None,
        target_timestamp=target.timestamp if target is not None else None,
        target_server_uuid=target.server_uuid if target is not None else None,
        target_gtid_executed=target.gtid_executed if target is not None else None,
        target_set=target.gtid_set if target is not None else None,
        frames_scanned=frames_scanned,
        raw_delay_seconds=raw_delay_seconds,
    )


def _extract_observation(
    replay_id: int, timestamp: str, data: dict[str, Any], *, side: str
) -> tuple[FrameObservation | None, str | None]:
    """Builds a normalized observation from a frame's payload.

    Returns (observation, None) or (None, reason). ``side`` names the recording in the
    message. The server-level set comes from ``global_variables.gtid_executed`` only;
    per-channel replication rows are never consulted.
    """
    variables = data.get("global_variables")
    if not isinstance(variables, dict):
        return None, f"{side} frame {replay_id} is missing global_variables"

    gtid_executed = variables.get("gtid_executed")
    if gtid_executed is None:
        return None, f"{side} frame {replay_id} is missing gtid_executed (GTID not enabled or not recorded)"
    if not isinstance(gtid_executed, str) or not gtid_executed.strip():
        return None, f"{side} frame {replay_id} has an empty gtid_executed (GTID is not enabled)"

    server_uuid = variables.get("server_uuid")
    if not isinstance(server_uuid, str) or not server_uuid.strip():
        return None, f"{side} frame {replay_id} is missing server_uuid"

    try:
        gtid_set = parse_gtid_set(gtid_executed)
    except GtidTaggedError as cause:
        return None, f"{side} frame {replay_id} uses tagged GTIDs, which are outside the supported scope ({cause})"
    except GtidParseError as cause:
        return None, f"{side} frame {replay_id} has an unparseable gtid_executed: {cause}"

    return (
        FrameObservation(
            replay_id=replay_id,
            timestamp=timestamp,
            server_uuid=server_uuid.strip().lower(),
            gtid_executed=gtid_executed.strip(),
            gtid_set=gtid_set,
        ),
        None,
    )


def _raw_delay_seconds(master_timestamp: str, target_timestamp: str) -> float | None:
    """Difference of the two raw UTC frame timestamps; None if either is unparseable."""
    try:
        # Row timestamps are UTC whole seconds by recorder contract; tag them as such
        master_dt = datetime.fromisoformat(f"{master_timestamp}+00:00")
        target_dt = datetime.fromisoformat(f"{target_timestamp}+00:00")
    except (ValueError, TypeError):
        return None
    return (target_dt - master_dt).total_seconds()


def locate_gtid_barrier(
    master_manager: ReplayManager,
    master_frame_id: int,
    target_manager: ReplayManager,
    segment_start_id: int,
    segment_end_id: int,
) -> BarrierMatchResult:
    """Scans the target segment for the first frame containing the master frame's set.

    Args:
        master_manager: Replay manager for the master recording.
        master_frame_id: Frame on the master recording whose gtid_executed is the barrier.
        target_manager: Replay manager for the replica recording.
        segment_start_id: First frame (inclusive) of the target segment.
        segment_end_id: Last frame (inclusive) of the target segment.

    Returns:
        BarrierMatchResult with status matched/not_reached/indeterminate and evidence.
        This function never moves either manager's playback cursor.
    """
    if segment_start_id <= 0 or segment_start_id > segment_end_id:
        return BarrierMatchResult(
            status=BarrierStatus.indeterminate,
            reason=f"Invalid target segment: frame {segment_start_id} to frame {segment_end_id}",
            segment_start_id=segment_start_id,
            segment_end_id=segment_end_id,
        )

    # --- The barrier: the master frame's complete server-level gtid_executed --------
    try:
        master_frame = master_manager.read_recorded_frame(master_frame_id)
    except Exception as cause:  # CorruptFrameError; surface deterministically
        return BarrierMatchResult(
            status=BarrierStatus.indeterminate,
            reason=f"Master frame {master_frame_id} is corrupt: {cause}",
        )
    if master_frame is None:
        return BarrierMatchResult(
            status=BarrierStatus.indeterminate,
            reason=f"Master frame {master_frame_id} is missing from the recording",
        )

    master_timestamp, master_data = master_frame
    master_observation, reason = _extract_observation(master_frame_id, master_timestamp, master_data, side="Master")
    if master_observation is None:
        return BarrierMatchResult(status=BarrierStatus.indeterminate, reason=reason)

    barrier = master_observation.gtid_set

    # --- Walk the target segment strictly in record order ---------------------------
    previous: FrameObservation | None = None
    last_observation: FrameObservation | None = None
    frames_scanned = 0

    for replay_id in range(segment_start_id, segment_end_id + 1):
        try:
            frame = target_manager.read_recorded_frame(replay_id)
        except Exception as cause:  # CorruptFrameError: do not cross a broken frame
            return _build_result(
                BarrierStatus.indeterminate,
                f"Target frame {replay_id} is corrupt, stopping before the barrier is proven: {cause}",
                master=master_observation,
                barrier=barrier,
                segment_start_id=segment_start_id,
                segment_end_id=segment_end_id,
                frames_scanned=frames_scanned,
            )
        if frame is None:
            return _build_result(
                BarrierStatus.indeterminate,
                f"Target frame {replay_id} is missing inside the segment; containment cannot be proven across the gap",
                master=master_observation,
                barrier=barrier,
                segment_start_id=segment_start_id,
                segment_end_id=segment_end_id,
                frames_scanned=frames_scanned,
            )

        timestamp, data = frame
        observation, reason = _extract_observation(replay_id, timestamp, data, side="Target")
        if observation is None:
            return _build_result(
                BarrierStatus.indeterminate,
                reason,
                master=master_observation,
                barrier=barrier,
                segment_start_id=segment_start_id,
                segment_end_id=segment_end_id,
                frames_scanned=frames_scanned,
            )

        frames_scanned += 1

        # A server_uuid change inside the segment is a generation break (re-provisioned
        # or restored data directory), not the same execution history.
        if previous is not None and observation.server_uuid != previous.server_uuid:
            return _build_result(
                BarrierStatus.indeterminate,
                f"Target server_uuid changed at frame {replay_id}: {previous.server_uuid} -> "
                f"{observation.server_uuid} (generation break), stopping",
                master=master_observation,
                barrier=barrier,
                segment_start_id=segment_start_id,
                segment_end_id=segment_end_id,
                frames_scanned=frames_scanned,
                target=observation,
            )

        # Executed sets only grow. An observed frame that fails to contain the frame
        # before it shows a reset/rollback (or a skipped stretch of history), so later
        # containment of the barrier would be meaningless.
        if previous is not None and not observation.gtid_set.contains(previous.gtid_set):
            return _build_result(
                BarrierStatus.indeterminate,
                f"Target gtid_executed at frame {replay_id} no longer contains the previous frame's set "
                "(an observed reset or rollback), stopping",
                master=master_observation,
                barrier=barrier,
                segment_start_id=segment_start_id,
                segment_end_id=segment_end_id,
                frames_scanned=frames_scanned,
                target=observation,
            )

        # First frame (in record order) that contains the whole barrier is the answer.
        if observation.gtid_set.contains(barrier):
            return _build_result(
                BarrierStatus.matched,
                None,
                master=master_observation,
                barrier=barrier,
                segment_start_id=segment_start_id,
                segment_end_id=segment_end_id,
                frames_scanned=frames_scanned,
                target=observation,
                raw_delay_seconds=_raw_delay_seconds(master_observation.timestamp, observation.timestamp),
            )

        previous = observation
        last_observation = observation

    # Valid segment, monotonic chain, but the barrier was still not covered at the end.
    return _build_result(
        BarrierStatus.not_reached,
        f"The barrier is not contained by the target's gtid_executed through frame {segment_end_id}; "
        "the recorded segment ends before the master frame's transactions were observed",
        master=master_observation,
        barrier=barrier,
        segment_start_id=segment_start_id,
        segment_end_id=segment_end_id,
        frames_scanned=frames_scanned,
        target=last_observation,
    )
