from __future__ import annotations

import pytest

from dolphie.Modules.GtidSet import GtidParseError, GtidTaggedError, parse_gtid_set

UUID_A = "3e11fa47-71ca-11e1-9e33-c80aa9429562"
UUID_B = "aaaaaaaa-71ca-11e1-9e33-c80aa9429562"


def test_parses_single_numbers_ranges_and_gaps() -> None:
    parsed = parse_gtid_set(f"{UUID_A}:1-10:20-30:47")
    intervals = parsed.entries[UUID_A]
    assert [(interval.start, interval.end) for interval in intervals] == [(1, 10), (20, 30), (47, 47)]


def test_parses_multiple_uuids_newlines_whitespace_and_normalizes_case() -> None:
    parsed = parse_gtid_set(f"  {UUID_A.upper()}:1-3,\n  {UUID_B}:5  ")
    assert set(parsed.entries) == {UUID_A, UUID_B}
    assert parsed.entries[UUID_A][0].start == 1


def test_merges_duplicate_and_adjacent_entries_for_one_uuid() -> None:
    parsed = parse_gtid_set(f"{UUID_A}:1-5:6-10,{UUID_A}:8-20")
    assert [(interval.start, interval.end) for interval in parsed.entries[UUID_A]] == [(1, 20)]


@pytest.mark.parametrize(
    "text",
    [
        None,
        "",
        "   ",
        f"{UUID_A}",
        f"{UUID_A}:",
        "not-a-uuid:1-10",
        "ANONYMOUS:1-10",
        f"{UUID_A}:10-5",
        f"{UUID_A}:abc",
        f"{UUID_A}:1-",
        f"{UUID_A}:1-:",
    ],
)
def test_malformed_sets_are_rejected(text: object) -> None:
    with pytest.raises(GtidParseError):
        parse_gtid_set(text)


def test_a_uuid_with_the_wrong_layout_is_rejected() -> None:
    with pytest.raises(GtidParseError):
        parse_gtid_set(f"{UUID_A[:-1]}g:1-10")


def test_tagged_gtids_are_rejected_as_out_of_scope() -> None:
    # MySQL tagged form uuid:tag:intervals must never be mistaken for uuid:interval
    with pytest.raises(GtidTaggedError):
        parse_gtid_set(f"{UUID_A}:shard1:1-10")
    with pytest.raises(GtidTaggedError):
        parse_gtid_set(f"{UUID_A}:1-5,{UUID_B}:tag:6")


def test_contains_handles_equality_extra_uuids_and_gap_coverage() -> None:
    base = parse_gtid_set(f"{UUID_A}:1-10:20-30")

    assert base.contains(parse_gtid_set(f"{UUID_A}:1-10:20-30"))
    assert parse_gtid_set(f"{UUID_A}:1-30,{UUID_B}:1-9").contains(base)
    # The target must cover the gaps too
    assert not parse_gtid_set(f"{UUID_A}:1-25").contains(base)
    assert not parse_gtid_set(f"{UUID_A}:1-10:21-30").contains(base)
    assert not parse_gtid_set(f"{UUID_B}:1-30").contains(base)


def test_contains_works_with_single_transaction_boundaries() -> None:
    outer = parse_gtid_set(f"{UUID_A}:9")
    assert outer.contains(parse_gtid_set(f"{UUID_A}:9"))
    assert not outer.contains(parse_gtid_set(f"{UUID_A}:10"))


def test_contains_accepts_sixty_four_bit_transaction_numbers() -> None:
    max_trx = 2**64 - 1
    outer = parse_gtid_set(f"{UUID_A}:1-{max_trx}")
    inner = parse_gtid_set(f"{UUID_A}:{max_trx}")
    assert outer.contains(inner)
    assert not parse_gtid_set(f"{UUID_A}:1-{max_trx - 1}").contains(inner)
