"""Parser and containment check for the MySQL UUID GTID sets used by replay comparison.

Only the untagged MySQL form is in scope: comma/newline-separated entries of
``uuid:interval[:interval]...``, where every interval is a single transaction number
or an inclusive ``start-end`` range. A set may carry several UUIDs and each UUID may
carry disjoint intervals (for example ``uuid:1-10:20-30``). Tagged GTIDs
(``uuid:tag:interval`` from newer MySQL versions) are rejected on sight rather than
silently misread, because the extra colon makes the tag look like an interval.

The module deliberately proves nothing but set containment between two observations:
it never infers that row data is identical or that a reset between two samples did not
happen.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# A MySQL server UUID is exactly 32 hexadecimal digits in the canonical 8-4-4-4-12
# layout. Anything else (ANONYMOUS, a tag, free text) is out of scope.
_UUID_PATTERN = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_INTERVAL_PATTERN = re.compile(r"^(\d+)(?:-(\d+))?$")


class GtidParseError(ValueError):
    """Raised when text cannot be parsed as an untagged MySQL UUID GTID set."""


class GtidTaggedError(GtidParseError):
    """Raised when the text uses tagged GTIDs, which are outside the supported scope."""


@dataclass(frozen=True)
class GtidInterval:
    """An inclusive interval of transaction numbers for one UUID."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 1 or self.end < self.start:
            raise GtidParseError(f"Invalid GTID interval {self.start}-{self.end}")


@dataclass(frozen=True)
class GtidSet:
    """A parsed GTID set: UUID to normalized, sorted, merged intervals."""

    # The original text, kept so the UI displays exactly what the frame recorded.
    raw: str
    entries: dict[str, tuple[GtidInterval, ...]]

    @property
    def uuids(self) -> tuple[str, ...]:
        return tuple(self.entries)

    def contains(self, other: GtidSet) -> bool:
        """Whether every transaction in ``other`` is present in this set.

        Each UUID of ``other`` must exist here, and each of its intervals must be fully
        covered by the union of this set's intervals for that UUID. Gaps inside either
        side are honored, so ``uuid:1-10:20-30`` is not contained in ``uuid:1-25``.
        """
        for uuid, other_intervals in other.entries.items():
            own_intervals = self.entries.get(uuid)
            if own_intervals is None:
                return False
            for interval in other_intervals:
                if not _intervals_cover(own_intervals, interval):
                    return False
        return True

    def normalized_text(self) -> str:
        """Stable rendering of the parsed set, lowercased with merged intervals."""
        lines: list[str] = []
        for uuid, intervals in self.entries.items():
            rendered = ":".join(
                str(interval.start) if interval.start == interval.end else f"{interval.start}-{interval.end}"
                for interval in intervals
            )
            lines.append(f"{uuid}:{rendered}")
        return ",".join(lines)


def _intervals_cover(intervals: tuple[GtidInterval, ...], target: GtidInterval) -> bool:
    """Whether the merged ``intervals`` cover every number in the inclusive ``target``."""
    remaining_start = target.start
    for interval in intervals:
        if interval.end < remaining_start:
            continue
        if interval.start > remaining_start:
            # A gap before the next interval leaves part of the target uncovered.
            return False
        remaining_start = max(remaining_start, interval.end + 1)
        if remaining_start > target.end:
            return True
    return False


def _parse_interval(text: str, *, context: str) -> GtidInterval:
    match = _INTERVAL_PATTERN.match(text)
    if not match:
        # A non-numeric token where the intervals go is the tagged-GTID shape
        # (uuid:tag:1-10); call it out specifically instead of a generic parse error.
        raise GtidTaggedError(f"Tagged or non-numeric GTID component {text!r} in {context!r}")

    start = int(match.group(1))
    end = int(match.group(2)) if match.group(2) is not None else start
    try:
        return GtidInterval(start, end)
    except GtidParseError:
        raise


def _merge_intervals(intervals: list[GtidInterval]) -> tuple[GtidInterval, ...]:
    ordered = sorted(intervals, key=lambda interval: interval.start)
    merged: list[GtidInterval] = []
    for interval in ordered:
        if merged and interval.start <= merged[-1].end + 1:
            merged[-1] = GtidInterval(merged[-1].start, max(merged[-1].end, interval.end))
        else:
            merged.append(interval)
    return tuple(merged)


def parse_gtid_set(value: object) -> GtidSet:
    """Parse an untagged MySQL UUID GTID set into a normalized :class:`GtidSet`.

    Args:
        value: The recorded ``gtid_executed`` text. Non-string or empty input raises.

    Raises:
        GtidParseError: Empty/whitespace input, a malformed UUID, a malformed interval,
            or unparseable text.
        GtidTaggedError: The entry carries a GTID tag, which is outside the scope here.
    """
    if not isinstance(value, str):
        raise GtidParseError(f"GTID set must be a string, got {type(value).__name__}")

    raw = value.strip()
    if not raw:
        raise GtidParseError("GTID set is empty")

    entries: dict[str, list[GtidInterval]] = {}
    # SHOW output wraps lines as ",\n", so a comma and a newline can sit back to back.
    # Adjacent separators and blank lines carry no entry and are skipped.
    parts = [part.strip() for part in re.split(r"[\n,]", raw)]
    for entry in parts:
        if not entry:
            continue

        components = [component.strip() for component in entry.split(":")]
        if len(components) < 2:
            raise GtidParseError(f"GTID entry {entry!r} has no transaction intervals")

        uuid = components[0].lower()
        if not _UUID_PATTERN.match(uuid):
            raise GtidParseError(f"GTID entry {entry!r} does not start with a MySQL server UUID")

        intervals = [_parse_interval(component, context=entry) for component in components[1:]]
        entries.setdefault(uuid, []).extend(intervals)

    return GtidSet(raw=raw, entries={uuid: _merge_intervals(intervals) for uuid, intervals in entries.items()})
