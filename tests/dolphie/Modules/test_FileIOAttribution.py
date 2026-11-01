"""Controlled continuous-sampling tests for the InnoDB file I/O attribution ledger.

No database is involved: every "sample" is a canned tuple of file_summary_by_instance,
INNODB_TABLESPACES and INNODB_TABLES rows fed consecutively into the ledger. The
invariants under test are: each computable non-negative increment booked exactly once,
groups summing to the physical increments, immovable history across mapping changes,
shared bytes kept at the space group, whole-file baseline rebuild on any regression, and
missing measurements kept distinct from measured-but-unattributed bytes.
"""

from __future__ import annotations

import pytest

from dolphie.Modules.FileIOAttribution import (
    OBJECT_KEY_PREFIX,
    SPACE_KEY_PREFIX,
    UNATTRIBUTED_KEY,
    FileIOAttributionLedger,
    parse_object_name,
    resolve_tablespace_path,
)

DATADIR = "/var/lib/mysql/"


def file_row(path: str, read: int | None, write: int | None, *, absolute: bool = False) -> dict:
    file_name = path if absolute else f"{DATADIR}{path}"
    return {
        "FILE_NAME": file_name,
        "SUM_NUMBER_OF_BYTES_READ": read,
        "SUM_NUMBER_OF_BYTES_WRITE": write,
    }


def space_row(space: int, name: str, path: str, space_type: str = "Single") -> dict:
    return {"SPACE": space, "NAME": name, "PATH": path, "SPACE_TYPE": space_type}


def table_row(table_id: int, name: str, space: int) -> dict:
    return {"TABLE_ID": table_id, "NAME": name, "SPACE": space}


# The identity universe most scenarios use.
def base_identity() -> tuple[list[dict], list[dict]]:
    spaces = [
        space_row(10, "db/t1", "./db/t1.ibd"),
        space_row(11, "db/tp#P#p0", "./db/tp#P#p0.ibd"),
        space_row(20, "gen", "./gen.ibd", "General"),
        space_row(0, "ibdata1", "./ibdata1", "System"),
        space_row(4294967279, "innodb_undo_001", "./undo_001", "Undo"),
    ]
    tables = [
        table_row(100, "db/t1", 10),
        table_row(110, "db/tp#P#p0", 11),
        table_row(1, "db/a", 20),
        table_row(2, "db/b", 20),
    ]
    return spaces, tables


def record(ledger: FileIOAttributionLedger, files: list[dict], spaces: list[dict], tables: list[dict]):
    return ledger.record_sample(files, spaces, tables, DATADIR)


def assert_conservation(ledger: FileIOAttributionLedger) -> None:
    grouped_read = sum(entry.read_bytes for entry in ledger.entries.values())
    grouped_write = sum(entry.write_bytes for entry in ledger.entries.values())
    assert grouped_read == ledger.total_computed_read
    assert grouped_write == ledger.total_computed_write
    assert ledger.conservation_holds()


def test_first_sample_only_anchors_baselines_and_books_nothing() -> None:
    ledger = FileIOAttributionLedger()
    spaces, tables = base_identity()
    files = [file_row("db/t1.ibd", 100, 50)]

    result = record(ledger, files, spaces, tables)

    assert result.sample_index == 1
    assert result.new_baseline_files == [f"{DATADIR}db/t1.ibd"]
    assert result.computed_read == result.computed_write == 0
    assert ledger.total_computed_read == ledger.total_computed_write == 0
    assert ledger.entries[UNATTRIBUTED_KEY].read_bytes == 0
    assert_conservation(ledger)


def test_stable_single_tablespace_attributes_to_table() -> None:
    ledger = FileIOAttributionLedger()
    spaces, tables = base_identity()

    r1 = record(ledger, [file_row("db/t1.ibd", 100, 50)], spaces, tables)
    r2 = record(ledger, [file_row("db/t1.ibd", 140, 90)], spaces, tables)
    r3 = record(ledger, [file_row("db/t1.ibd", 200, 90)], spaces, tables)

    assert r1.computed_read == 0
    assert (r2.computed_read, r2.computed_write) == (40, 40)
    assert (r3.computed_read, r3.computed_write) == (60, 0)

    entry = ledger.entries[f"{OBJECT_KEY_PREFIX}100:db/t1"]
    assert (entry.read_bytes, entry.write_bytes) == (100, 40)
    assert (entry.last_read_bytes, entry.last_write_bytes) == (60, 0)
    assert entry.kind == "table"
    assert ledger.unattributed_read == 0
    assert_conservation(ledger)


def test_partitions_are_distinct_objects() -> None:
    ledger = FileIOAttributionLedger()
    spaces = [
        space_row(11, "db/tp#P#p0", "./db/tp#P#p0.ibd"),
        space_row(12, "db/tp#P#p1", "./db/tp#P#p1.ibd"),
    ]
    tables = [table_row(110, "db/tp#P#p0", 11), table_row(111, "db/tp#P#p1", 12)]

    files = [file_row("db/tp#P#p0.ibd", 0, 0), file_row("db/tp#P#p1.ibd", 0, 0)]
    record(ledger, files, spaces, tables)
    files = [file_row("db/tp#P#p0.ibd", 30, 3), file_row("db/tp#P#p1.ibd", 70, 7)]
    record(ledger, files, spaces, tables)

    p0 = ledger.entries[f"{OBJECT_KEY_PREFIX}110:db/tp#P#p0"]
    p1 = ledger.entries[f"{OBJECT_KEY_PREFIX}111:db/tp#P#p1"]
    assert p0.kind == p1.kind == "partition"
    assert (p0.read_bytes, p0.write_bytes) == (30, 3)
    assert (p1.read_bytes, p1.write_bytes) == (70, 7)
    assert_conservation(ledger)


def test_shared_general_tablespace_booked_once_not_copied_per_table() -> None:
    ledger = FileIOAttributionLedger()
    spaces = [space_row(20, "gen", "./gen.ibd", "General")]
    tables = [table_row(1, "db/a", 20), table_row(2, "db/b", 20)]

    record(ledger, [file_row("gen.ibd", 0, 0)], spaces, tables)
    record(ledger, [file_row("gen.ibd", 300, 400)], spaces, tables)

    space_entries = [key for key in ledger.entries if key.startswith(SPACE_KEY_PREFIX)]
    assert space_entries == [f"{SPACE_KEY_PREFIX}20:gen"]
    entry = ledger.entries[space_entries[0]]
    assert entry.kind == "space"
    assert (entry.read_bytes, entry.write_bytes) == (300, 400)
    assert f"{OBJECT_KEY_PREFIX}1:db/a" not in ledger.entries
    assert f"{OBJECT_KEY_PREFIX}2:db/b" not in ledger.entries
    assert_conservation(ledger)


def test_system_and_undo_spaces_without_members_are_shared_groups() -> None:
    ledger = FileIOAttributionLedger()
    spaces = [
        space_row(0, "ibdata1", "./ibdata1", "System"),
        space_row(4294967279, "innodb_undo_001", "./undo_001", "Undo"),
    ]

    files = [
        file_row("ibdata1", 0, 0),
        file_row("undo_001", 0, 0),
    ]
    record(ledger, files, spaces, [])
    files = [
        file_row("ibdata1", 11, 22),
        file_row("undo_001", 33, 44),
    ]
    record(ledger, files, spaces, [])

    assert ledger.entries[f"{SPACE_KEY_PREFIX}0:ibdata1"].read_bytes == 11
    assert ledger.entries[f"{SPACE_KEY_PREFIX}0:ibdata1"].write_bytes == 22
    assert (ledger.entries[f"{SPACE_KEY_PREFIX}4294967279:innodb_undo_001"].read_bytes) == 33
    assert ledger.entries[f"{SPACE_KEY_PREFIX}4294967279:innodb_undo_001"].write_bytes == 44
    assert_conservation(ledger)


def test_absolute_datafile_path_matches_exactly() -> None:
    ledger = FileIOAttributionLedger()
    spaces = [space_row(21, "ext", "/data/external.ibd", "General")]

    record(ledger, [file_row("/data/external.ibd", 0, 0, absolute=True)], spaces, [])
    result = record(ledger, [file_row("/data/external.ibd", 5, 6, absolute=True)], spaces, [])

    assert result.booked_read == 5
    assert f"{SPACE_KEY_PREFIX}21:ext" in ledger.entries
    assert_conservation(ledger)


def test_file_without_identity_is_unattributed_not_missing() -> None:
    ledger = FileIOAttributionLedger()
    spaces, tables = base_identity()

    record(ledger, [file_row("/var/log/mysql/binlog.000001", 0, 0, absolute=True)], spaces, tables)
    result = record(ledger, [file_row("/var/log/mysql/binlog.000001", 50, 60, absolute=True)], spaces, tables)

    assert result.missing_files == []
    assert ledger.missing_intervals == 0
    assert (ledger.unattributed_read, ledger.unattributed_write) == (50, 60)
    assert ledger.unknown_identity_unattributed_read == 50
    assert ledger.unknown_identity_unattributed_write == 60
    assert_conservation(ledger)


def test_ambiguous_path_is_unattributed_without_guessing() -> None:
    ledger = FileIOAttributionLedger()
    spaces = [
        space_row(30, "db/a", "./dup.ibd"),
        space_row(31, "db/b", "./dup.ibd", "General"),
    ]

    record(ledger, [file_row("dup.ibd", 0, 0)], spaces, [])
    result = record(ledger, [file_row("dup.ibd", 9, 8)], spaces, [])

    assert result.booked_read == 9
    assert ledger.unattributed_read == 9
    assert not any(key.startswith(OBJECT_KEY_PREFIX) for key in ledger.entries)
    assert_conservation(ledger)


def test_mapping_change_interval_is_unattributed_then_new_mapping_books() -> None:
    """Partition exchange: the file path is stable while its table identity swaps."""
    ledger = FileIOAttributionLedger()

    spaces_before = [space_row(11, "db/tp#P#p0", "./db/tp#P#p0.ibd")]
    tables_before = [table_row(110, "db/tp#P#p0", 11)]
    spaces_after = [space_row(11, "db/standalone", "./db/tp#P#p0.ibd")]
    tables_after = [table_row(220, "db/standalone", 11)]

    record(ledger, [file_row("db/tp#P#p0.ibd", 0, 0)], spaces_before, tables_before)
    record(ledger, [file_row("db/tp#P#p0.ibd", 100, 0)], spaces_before, tables_before)
    transition = record(ledger, [file_row("db/tp#P#p0.ibd", 150, 30)], spaces_after, tables_after)
    settled = record(ledger, [file_row("db/tp#P#p0.ibd", 160, 60)], spaces_after, tables_after)

    assert transition.transition_files == [f"{DATADIR}db/tp#P#p0.ibd"]
    assert (transition.booked_read, transition.booked_write) == (50, 30)
    assert (ledger.transition_unattributed_read, ledger.transition_unattributed_write) == (50, 30)
    assert (settled.booked_read, settled.booked_write) == (10, 30)

    # Old partition history stays exactly where it was booked; it is never moved.
    assert ledger.entries[f"{OBJECT_KEY_PREFIX}110:db/tp#P#p0"].read_bytes == 100
    new_object = ledger.entries[f"{OBJECT_KEY_PREFIX}220:db/standalone"]
    assert (new_object.read_bytes, new_object.write_bytes) == (10, 30)
    assert (ledger.unattributed_read, ledger.unattributed_write) == (50, 30)
    assert_conservation(ledger)


def test_rename_path_swap_keeps_old_history_and_baselines_new_path() -> None:
    ledger = FileIOAttributionLedger()

    spaces_before = [space_row(10, "db/t1", "./db/t1.ibd")]
    tables_before = [table_row(100, "db/t1", 10)]
    spaces_after = [space_row(10, "db/t1r", "./db/t1r.ibd")]
    tables_after = [table_row(100, "db/t1r", 10)]

    record(ledger, [file_row("db/t1.ibd", 100, 100)], spaces_before, tables_before)
    record(ledger, [file_row("db/t1.ibd", 140, 140)], spaces_before, tables_before)
    rename_interval = record(ledger, [file_row("db/t1r.ibd", 0, 0)], spaces_after, tables_after)
    record(ledger, [file_row("db/t1r.ibd", 40, 10)], spaces_after, tables_after)

    assert f"{DATADIR}db/t1.ibd" in rename_interval.missing_files
    assert f"{DATADIR}db/t1r.ibd" in rename_interval.new_baseline_files
    assert rename_interval.computed_read == rename_interval.computed_write == 0

    assert ledger.entries[f"{OBJECT_KEY_PREFIX}100:db/t1"].read_bytes == 40
    assert ledger.entries[f"{OBJECT_KEY_PREFIX}100:db/t1r"].write_bytes == 10
    assert ledger.unattributed_read == 0
    assert_conservation(ledger)


def test_drop_recreate_path_reuse_does_not_transfer_history() -> None:
    ledger = FileIOAttributionLedger()

    spaces_v1 = [space_row(30, "db/x", "./db/x.ibd")]
    tables_v1 = [table_row(300, "db/x", 30)]
    spaces_v2 = [space_row(31, "db/x", "./db/x.ibd")]
    tables_v2 = [table_row(400, "db/x", 31)]

    record(ledger, [file_row("db/x.ibd", 1000, 2000)], spaces_v1, tables_v1)
    record(ledger, [file_row("db/x.ibd", 1100, 2050)], spaces_v1, tables_v1)
    dropped = record(ledger, [], spaces_v2, tables_v2)
    recreated = record(ledger, [file_row("db/x.ibd", 0, 0)], spaces_v2, tables_v2)
    settled = record(ledger, [file_row("db/x.ibd", 70, 20)], spaces_v2, tables_v2)

    assert f"{DATADIR}db/x.ibd" in dropped.missing_files
    assert recreated.rollback_files == [f"{DATADIR}db/x.ibd"]
    assert (settled.booked_read, settled.booked_write) == (70, 20)

    old_generation = ledger.entries[f"{OBJECT_KEY_PREFIX}300:db/x"]
    new_generation = ledger.entries[f"{OBJECT_KEY_PREFIX}400:db/x"]
    assert (old_generation.read_bytes, old_generation.write_bytes) == (100, 50)
    assert (new_generation.read_bytes, new_generation.write_bytes) == (70, 20)
    assert_conservation(ledger)


def test_shared_space_membership_change_transitions_then_rebooks_to_space() -> None:
    ledger = FileIOAttributionLedger()

    spaces = [space_row(20, "gen", "./gen.ibd", "General")]
    tables_v1 = [table_row(1, "db/a", 20), table_row(2, "db/b", 20)]
    tables_v2 = [table_row(1, "db/a", 20), table_row(2, "db/b", 20), table_row(3, "db/c", 20)]

    record(ledger, [file_row("gen.ibd", 0, 0)], spaces, tables_v1)
    record(ledger, [file_row("gen.ibd", 300, 400)], spaces, tables_v1)
    transition = record(ledger, [file_row("gen.ibd", 350, 410)], spaces, tables_v2)
    settled = record(ledger, [file_row("gen.ibd", 360, 420)], spaces, tables_v2)

    assert transition.transition_files
    assert (ledger.unattributed_read, ledger.unattributed_write) == (50, 10)
    assert (settled.booked_read, settled.booked_write) == (10, 10)
    entry = ledger.entries[f"{SPACE_KEY_PREFIX}20:gen"]
    assert (entry.read_bytes, entry.write_bytes) == (310, 410)
    assert not any(key.startswith(OBJECT_KEY_PREFIX) for key in ledger.entries)
    assert_conservation(ledger)


def test_counter_regression_rebuilds_whole_file_baseline() -> None:
    ledger = FileIOAttributionLedger()
    spaces, tables = base_identity()

    def sample(t1_read: int | None, t1_write: int | None, other_read: int | None, other_write: int | None):
        files = [
            file_row("db/t1.ibd", t1_read, t1_write),
            file_row("gen.ibd", other_read, other_write),
        ]
        return record(ledger, files, spaces, tables)

    sample(100, 100, 0, 0)
    # Read regressed; the +50 write increment of the same file is not booked either.
    regressed = sample(80, 150, 10, 10)
    # A different, stable file still books its own increment in the same interval.
    assert regressed.rollback_files == [f"{DATADIR}db/t1.ibd"]
    assert regressed.computed_read == regressed.computed_write == 10
    assert (regressed.booked_read, regressed.booked_write) == (10, 10)

    settled = sample(120, 170, 15, 30)
    # t1 books against the rebuilt baseline (80, 150): 40/20; the other file adds 5/20.
    assert (settled.booked_read, settled.booked_write) == (45, 40)
    t1 = ledger.entries[f"{OBJECT_KEY_PREFIX}100:db/t1"]
    assert (t1.read_bytes, t1.write_bytes) == (40, 20)
    assert ledger.rollback_events == 1
    assert_conservation(ledger)


def test_write_counter_regression_also_rebuilds_whole_file() -> None:
    ledger = FileIOAttributionLedger()
    spaces, tables = base_identity()

    record(ledger, [file_row("db/t1.ibd", 100, 100)], spaces, tables)
    result = record(ledger, [file_row("db/t1.ibd", 140, 90)], spaces, tables)

    assert result.rollback_files
    assert result.computed_read == result.computed_write == 0
    assert f"{OBJECT_KEY_PREFIX}100:db/t1" not in ledger.entries
    assert ledger.rollback_events == 1


def test_missing_counter_is_a_missing_measurement_never_zero_filled() -> None:
    ledger = FileIOAttributionLedger()
    spaces, tables = base_identity()

    record(ledger, [file_row("db/t1.ibd", 100, 100)], spaces, tables)
    missing = record(ledger, [file_row("db/t1.ibd", 130, None)], spaces, tables)
    assert missing.missing_files
    assert ledger.missing_intervals == 1
    assert ledger.unattributed_read == 0
    assert f"{OBJECT_KEY_PREFIX}100:db/t1" not in ledger.entries

    # Baseline was rebuilt at (130, None): the write side still lacks an anchor.
    still_missing = record(ledger, [file_row("db/t1.ibd", 150, 160)], spaces, tables)
    assert still_missing.missing_files
    assert ledger.missing_intervals == 2

    settled = record(ledger, [file_row("db/t1.ibd", 160, 180)], spaces, tables)
    assert (settled.computed_read, settled.computed_write) == (10, 20)
    entry = ledger.entries[f"{OBJECT_KEY_PREFIX}100:db/t1"]
    assert (entry.read_bytes, entry.write_bytes) == (10, 20)
    assert_conservation(ledger)


def test_failed_poll_in_between_still_uses_adjacent_successes() -> None:
    ledger = FileIOAttributionLedger()
    spaces, tables = base_identity()

    record(ledger, [file_row("db/t1.ibd", 100, 0)], spaces, tables)
    # No record_sample() call for the failed poll: counters stay cumulative.
    result = record(ledger, [file_row("db/t1.ibd", 250, 40)], spaces, tables)

    assert (result.computed_read, result.computed_write) == (150, 40)
    assert result.missing_files == []
    assert_conservation(ledger)


def test_unobserved_ddl_between_samples_is_never_reconstructed() -> None:
    ledger = FileIOAttributionLedger()
    spaces, tables = base_identity()

    record(ledger, [file_row("db/t1.ibd", 0, 0)], spaces, tables)
    # A -> (unobserved B that existed only inside the gap) -> A with a large jump.
    result = record(ledger, [file_row("db/t1.ibd", 1000, 1000)], spaces, tables)

    assert result.transition_files == []
    assert ledger.transition_intervals == 0
    assert (result.booked_read, result.booked_write) == (1000, 1000)
    assert_conservation(ledger)


def test_conservation_holds_across_a_mixed_controlled_sequence() -> None:
    ledger = FileIOAttributionLedger()
    spaces, tables = base_identity()

    sequences = [
        # sample 1: baselines only
        [file_row("db/t1.ibd", 0, 0), file_row("gen.ibd", 0, 0), file_row("ibdata1", 0, 0)],
        # sample 2: stable growth for three different groups
        [file_row("db/t1.ibd", 100, 10), file_row("gen.ibd", 50, 5), file_row("ibdata1", 7, 1)],
        # sample 3: zero increments, plus a brand-new file (baseline only)
        [
            file_row("db/t1.ibd", 100, 10),
            file_row("gen.ibd", 50, 5),
            file_row("ibdata1", 7, 1),
            file_row("db/new.ibd", 0, 0),
        ],
        # sample 4: t1 regresses; new file grows into its own unknown bucket (no space yet)
        [
            file_row("db/t1.ibd", 90, 12),
            file_row("gen.ibd", 60, 5),
            file_row("ibdata1", 7, 3),
            file_row("db/new.ibd", 4, 2),
        ],
    ]

    results = []
    for files in sequences:
        result = record(ledger, files, spaces, tables)
        results.append(result)
        assert result.read_conserved
        assert result.write_conserved
        assert_conservation(ledger)

    # sample 4 expected booking: gen +10/0, system +0/2, new unknown +4/2; t1 rebuilt.
    assert (results[3].booked_read, results[3].booked_write) == (14, 4)
    assert ledger.rollback_events == 1

    # Grand total: objects 100/10, shared general 60/5 + system 7/3 = 67/8, unattributed 4/2.
    object_totals = (
        sum(e.read_bytes for k, e in ledger.entries.items() if k.startswith(OBJECT_KEY_PREFIX)),
        sum(e.write_bytes for k, e in ledger.entries.items() if k.startswith(OBJECT_KEY_PREFIX)),
    )
    space_totals = (
        sum(e.read_bytes for k, e in ledger.entries.items() if k.startswith(SPACE_KEY_PREFIX)),
        sum(e.write_bytes for k, e in ledger.entries.items() if k.startswith(SPACE_KEY_PREFIX)),
    )
    assert object_totals == (100, 10)
    assert space_totals == (67, 8)
    assert (ledger.unattributed_read, ledger.unattributed_write) == (4, 2)
    assert (ledger.total_computed_read, ledger.total_computed_write) == (171, 20)
    assert_conservation(ledger)


def test_reset_starts_a_fresh_observation() -> None:
    ledger = FileIOAttributionLedger()
    spaces, tables = base_identity()

    record(ledger, [file_row("db/t1.ibd", 0, 0)], spaces, tables)
    record(ledger, [file_row("db/t1.ibd", 10, 10)], spaces, tables)
    assert ledger.samples == 2
    assert ledger.entries[f"{OBJECT_KEY_PREFIX}100:db/t1"].read_bytes == 10

    ledger.reset()
    assert ledger.samples == 0
    assert list(ledger.entries) == [UNATTRIBUTED_KEY]
    assert ledger.total_computed_read == ledger.total_computed_write == 0
    assert ledger.missing_intervals == ledger.rollback_events == 0


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("sakila/actor", ("sakila.actor", None, None)),
        ("sakila/t#P#p0", ("sakila.t", "p0", None)),
        ("sakila/t#P#p0#SP#sp0", ("sakila.t", "p0", "sp0")),
    ],
)
def test_parse_object_name(name: str, expected: tuple[str, str | None, str | None]) -> None:
    assert parse_object_name(name) == expected


def test_resolve_tablespace_path_handles_relative_and_absolute() -> None:
    assert resolve_tablespace_path("./sakila/actor.ibd", "/var/lib/mysql/") == "/var/lib/mysql/sakila/actor.ibd"
    assert resolve_tablespace_path("/data/x.ibd", "/var/lib/mysql/") == "/data/x.ibd"
    assert resolve_tablespace_path("undo_001", "/var/lib/mysql/") == "/var/lib/mysql/undo_001"
