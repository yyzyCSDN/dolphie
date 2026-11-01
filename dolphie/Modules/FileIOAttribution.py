"""Attribution of MySQL 8 InnoDB physical file I/O bytes to tables/partitions.

The ledger is fed adjacent successful samples of:

* ``performance_schema.file_summary_by_instance`` rows (per-file cumulative counters),
* ``information_schema.INNODB_TABLESPACES`` rows (tablespace identity and path), and
* ``information_schema.INNODB_TABLES`` rows (table/partition identity and its space).

Only non-negative counter increments between two adjacent successful samples are bookable.
An increment is booked exactly once, to exactly one of:

* the single table/partition of a ``Single`` tablespace with exactly one observed member,
* the shared tablespace group (everything else with a confirmed space: general, system,
  undo and temporary spaces, and any space without exactly one observed member), or
* the unattributed bucket (the endpoint mappings changed across the interval, or the
  file has no confirmable InnoDB identity).

Bytes already booked are never moved: RENAME, partition exchange, drop/recreate and path
reuse only affect intervals observed *after* the endpoints agree on the new mapping.
A read/write counter regression rebuilds the whole file's baseline (nothing is booked
for that interval). Missing counters are "missing measurements" and are not booked
anywhere; they are kept separate from measured-but-unattributed bytes.
"""

from __future__ import annotations

import posixpath
from dataclasses import dataclass, field
from typing import Final, Literal

from dolphie.DataTypes import DatabaseRow
from dolphie.Modules.Functions import coerce_int, coerce_str

EntryKind = Literal["table", "partition", "space", "unattributed"]

PARTITION_MARKER: Final = "#P#"
SUBPARTITION_MARKER: Final = "#SP#"
OBJECT_KEY_PREFIX: Final = "object:"
SPACE_KEY_PREFIX: Final = "space:"
UNATTRIBUTED_KEY: Final = "__unattributed__"

# Fingerprint of a file's mapping at one sample endpoint:
# (space id, space name, resolved path, space type, ((table id, table name), ...))
# ``None`` means the file could not be matched to a tablespace at that endpoint.
Fingerprint = tuple[int, str, str, str, tuple[tuple[int, str], ...]] | None

_READ_FIELD = "SUM_NUMBER_OF_BYTES_READ"
_WRITE_FIELD = "SUM_NUMBER_OF_BYTES_WRITE"


@dataclass(frozen=True)
class TablespaceIdentity:
    space: int
    name: str
    path: str
    space_type: str


@dataclass(frozen=True)
class TableIdentity:
    table_id: int
    name: str
    space: int


@dataclass
class LedgerEntry:
    key: str
    kind: EntryKind
    name: str
    detail: str = ""
    read_bytes: int = 0
    write_bytes: int = 0
    last_read_bytes: int = 0
    last_write_bytes: int = 0


@dataclass
class _Baseline:
    read: int | None
    write: int | None
    fingerprint: Fingerprint


@dataclass
class SampleResult:
    """What happened while folding one successful sample into the ledger."""

    sample_index: int
    observed_files: int = 0
    computed_read: int = 0
    computed_write: int = 0
    booked_read: int = 0
    booked_write: int = 0
    missing_files: list[str] = field(default_factory=list)
    rollback_files: list[str] = field(default_factory=list)
    transition_files: list[str] = field(default_factory=list)
    new_baseline_files: list[str] = field(default_factory=list)

    @property
    def read_conserved(self) -> bool:
        return self.computed_read == self.booked_read

    @property
    def write_conserved(self) -> bool:
        return self.computed_write == self.booked_write


def parse_object_name(name: str) -> tuple[str, str | None, str | None]:
    """Split ``INNODB_TABLES.NAME`` (``schema/table`` or ``schema/table#P#p[#SP#s]``).

    Returns the dotted qualified name, the partition name when present, and the
    subpartition name when present. This only formats an observed identity; it never
    infers one.
    """
    subpartition = None
    partition = None
    base_name = name

    if SUBPARTITION_MARKER in base_name:
        base_name, _, subpartition = base_name.rpartition(SUBPARTITION_MARKER)
    if PARTITION_MARKER in base_name:
        base_name, _, partition = base_name.rpartition(PARTITION_MARKER)

    schema, separator, table = base_name.partition("/")
    qualified = f"{schema}.{table}" if separator else base_name
    return qualified, partition, subpartition


def normalize_server_path(path: str) -> str:
    """Normalize a path reported by the *server* without resolving client-side symlinks.

    Backslashes are folded to ``/`` so a Windows client compares Linux server paths
    consistently; no case folding or symlink resolution is attempted.
    """
    return posixpath.normpath(path.replace("\\", "/"))


def resolve_tablespace_path(path: str, datadir: str) -> str:
    """Resolve an ``INNODB_TABLESPACES.PATH`` (``./schema/t.ibd`` or absolute) exactly."""
    normalized = normalize_server_path(path)
    if normalized.startswith("/"):
        return normalized

    if normalized.startswith("./"):
        normalized = normalized[2:]

    base = normalize_server_path(datadir).rstrip("/")
    return normalize_server_path(f"{base}/{normalized}")


def _counter(value: object) -> int | None:
    """A counter is present only as a real integer. Never zero-fill NULLs or garbage."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


class FileIOAttributionLedger:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Drop the ledger and every file baseline, starting a new observation."""
        self.samples: int = 0
        self._baselines: dict[str, _Baseline] = {}
        self.entries: dict[str, LedgerEntry] = {
            UNATTRIBUTED_KEY: LedgerEntry(key=UNATTRIBUTED_KEY, kind="unattributed", name="Unattributed")
        }
        self.total_computed_read: int = 0
        self.total_computed_write: int = 0
        # Subtotals of the unattributed bucket, kept distinct for the view.
        self.transition_unattributed_read: int = 0
        self.transition_unattributed_write: int = 0
        self.unknown_identity_unattributed_read: int = 0
        self.unknown_identity_unattributed_write: int = 0
        self.missing_intervals: int = 0
        self.rollback_events: int = 0
        self.transition_intervals: int = 0
        self.last_result: SampleResult | None = None

    @property
    def unattributed_read(self) -> int:
        return self.entries[UNATTRIBUTED_KEY].read_bytes

    @property
    def unattributed_write(self) -> int:
        return self.entries[UNATTRIBUTED_KEY].write_bytes

    def record_sample(
        self,
        file_rows: list[DatabaseRow],
        tablespace_rows: list[DatabaseRow],
        table_rows: list[DatabaseRow],
        datadir: str,
    ) -> SampleResult:
        """Fold one successful sample into the ledger and return its interval result.

        A sample is "successful" only when the caller obtained every data set; callers
        skip this method on a failed query so a failed poll is a missing observation
        rather than fabricated zero counters.
        """
        self.samples += 1
        result = SampleResult(sample_index=self.samples)

        path_to_space, ambiguous_paths = self._build_tablespaces(tablespace_rows, datadir)
        members_by_space = self._build_members(table_rows)

        current_files: set[str] = set()
        for row in file_rows:
            instance = row.get("FILE_NAME")
            if instance is None:
                # Without the file instance identity the counters cannot belong to a
                # baseline; skipping is not a missing measurement of a known file.
                continue

            file_name = coerce_str(instance)
            current_files.add(file_name)
            result.observed_files += 1

            read = _counter(row.get(_READ_FIELD))
            write = _counter(row.get(_WRITE_FIELD))
            fingerprint = self._fingerprint(file_name, path_to_space, ambiguous_paths, members_by_space)
            self._fold_file(file_name, read, write, fingerprint, result)

        # Files present at the previous endpoint but absent now: their current counter
        # is unobservable. Baselines are kept so a reappearing path cannot inherit the
        # old observation silently; regression/transition rules handle its return.
        for file_name in set(self._baselines) - current_files:
            result.missing_files.append(file_name)
            self.missing_intervals += 1

        self.total_computed_read += result.computed_read
        self.total_computed_write += result.computed_write
        self.last_result = result
        return result

    def display_entries(self) -> list[LedgerEntry]:
        """Attributed targets by descending total bytes, with the unattributed row last."""
        attributed = [entry for key, entry in self.entries.items() if key != UNATTRIBUTED_KEY]
        attributed.sort(key=lambda entry: entry.read_bytes + entry.write_bytes, reverse=True)
        return attributed + [self.entries[UNATTRIBUTED_KEY]]

    def conservation_holds(self) -> bool:
        """Every computed non-negative increment is present exactly once across groups."""
        grouped_read = sum(entry.read_bytes for entry in self.entries.values())
        grouped_write = sum(entry.write_bytes for entry in self.entries.values())
        return grouped_read == self.total_computed_read and grouped_write == self.total_computed_write

    def _fold_file(
        self,
        file_name: str,
        read: int | None,
        write: int | None,
        fingerprint: Fingerprint,
        result: SampleResult,
    ) -> None:
        previous = self._baselines.get(file_name)
        if previous is None:
            # First sighting only anchors the baseline; pre-observation bytes are never booked.
            self._baselines[file_name] = _Baseline(read, write, fingerprint)
            result.new_baseline_files.append(file_name)
            return

        if read is None or write is None or previous.read is None or previous.write is None:
            # A counter is missing on one endpoint: the interval is uncomputable for the
            # whole file. Rebuild the baseline from what is observed; never zero-fill.
            result.missing_files.append(file_name)
            self.missing_intervals += 1
            self._baselines[file_name] = _Baseline(read, write, fingerprint)
            return

        if read < previous.read or write < previous.write:
            # Any counter regression rebuilds the whole file's baseline, both directions.
            result.rollback_files.append(file_name)
            self.rollback_events += 1
            self._baselines[file_name] = _Baseline(read, write, fingerprint)
            return

        delta_read = read - previous.read
        delta_write = write - previous.write
        result.computed_read += delta_read
        result.computed_write += delta_write

        if fingerprint != previous.fingerprint:
            # Endpoint mappings disagree (RENAME, exchange, drop/recreate, path reuse,
            # membership change, identity gained/lost): the transition interval is not
            # attributable to either endpoint's objects.
            result.transition_files.append(file_name)
            self.transition_intervals += 1
            self._book_unattributed(delta_read, delta_write, transition=True)
        elif fingerprint is None:
            # Both endpoints lack a confirmable identity: measured, but unattributable.
            self._book_unattributed(delta_read, delta_write, transition=False)
        elif self._is_single_table(fingerprint):
            table_id, table_name = fingerprint[4][0]
            kind: EntryKind = "partition" if PARTITION_MARKER in table_name else "table"
            # The observed (table id, name) pair is the object identity: a drop/recreate
            # reusing a name gets a fresh ledger row instead of inheriting old history.
            self._book_entry(
                f"{OBJECT_KEY_PREFIX}{table_id}:{table_name}",
                kind,
                table_name,
                str(table_id),
                delta_read,
                delta_write,
            )
        else:
            space_id, space_name, _path, space_type, _members = fingerprint
            self._book_entry(
                f"{SPACE_KEY_PREFIX}{space_id}:{space_name}",
                "space",
                space_name or f"SPACE #{space_id}",
                space_type,
                delta_read,
                delta_write,
            )

        result.booked_read += delta_read
        result.booked_write += delta_write
        self._baselines[file_name] = _Baseline(read, write, fingerprint)

    def _book_entry(
        self,
        key: str,
        kind: EntryKind,
        name: str,
        detail: str,
        delta_read: int,
        delta_write: int,
    ) -> None:
        entry = self.entries.get(key)
        if entry is None:
            entry = LedgerEntry(key=key, kind=kind, name=name, detail=detail)
            self.entries[key] = entry
        entry.read_bytes += delta_read
        entry.write_bytes += delta_write
        entry.last_read_bytes = delta_read
        entry.last_write_bytes = delta_write

    def _book_unattributed(self, delta_read: int, delta_write: int, *, transition: bool) -> None:
        entry = self.entries[UNATTRIBUTED_KEY]
        entry.read_bytes += delta_read
        entry.write_bytes += delta_write
        entry.last_read_bytes = delta_read
        entry.last_write_bytes = delta_write
        if transition:
            self.transition_unattributed_read += delta_read
            self.transition_unattributed_write += delta_write
        else:
            self.unknown_identity_unattributed_read += delta_read
            self.unknown_identity_unattributed_write += delta_write

    @staticmethod
    def _is_single_table(fingerprint: tuple[int, str, str, str, tuple[tuple[int, str], ...]]) -> bool:
        # SPACE_TYPE is server-reported; a hidden member simply produces zero members,
        # which can never be misattributed to another table.
        return fingerprint[3].lower() == "single" and len(fingerprint[4]) == 1

    @staticmethod
    def _build_tablespaces(
        rows: list[DatabaseRow], datadir: str
    ) -> tuple[dict[str, TablespaceIdentity], set[str]]:
        path_to_space: dict[str, TablespaceIdentity] = {}
        ambiguous_paths: set[str] = set()

        for row in rows:
            raw_path = row.get("PATH")
            if raw_path is None:
                continue

            space = coerce_int(row.get("SPACE"))
            resolved = resolve_tablespace_path(coerce_str(raw_path), datadir)
            identity = TablespaceIdentity(
                space=space,
                name=coerce_str(row.get("NAME")),
                path=resolved,
                space_type=coerce_str(row.get("SPACE_TYPE")),
            )

            existing = path_to_space.get(resolved)
            if existing is not None and existing.space != space:
                # Two spaces claim the same file path: identity is not confirmable.
                ambiguous_paths.add(resolved)
                continue
            path_to_space[resolved] = identity

        return path_to_space, ambiguous_paths

    @staticmethod
    def _build_members(rows: list[DatabaseRow]) -> dict[int, tuple[TableIdentity, ...]]:
        members: dict[int, list[TableIdentity]] = {}
        for row in rows:
            space = coerce_int(row.get("SPACE"))
            members.setdefault(space, []).append(
                TableIdentity(
                    table_id=coerce_int(row.get("TABLE_ID")),
                    name=coerce_str(row.get("NAME")),
                    space=space,
                )
            )
        return {
            space: tuple(sorted(group, key=lambda table: (table.table_id, table.name)))
            for space, group in members.items()
        }

    @staticmethod
    def _fingerprint(
        file_name: str,
        path_to_space: dict[str, TablespaceIdentity],
        ambiguous_paths: set[str],
        members_by_space: dict[int, tuple[TableIdentity, ...]],
    ) -> Fingerprint:
        normalized = normalize_server_path(file_name)
        if normalized in ambiguous_paths:
            return None

        space = path_to_space.get(normalized)
        if space is None:
            return None

        members = tuple(sorted((table.table_id, table.name) for table in members_by_space.get(space.space, ())))
        return space.space, space.name, space.path, space.space_type, members
