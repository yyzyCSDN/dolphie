"""Version-scoped hit history for ProxySQL ``runtime_mysql_query_rules``.

ProxySQL resets ``stats_mysql_query_rules.hits`` on every
``LOAD MYSQL QUERY RULES TO RUNTIME`` and reuses ``rule_id`` values freely, so keying a hit
rate by ``rule_id`` stitches unrelated rule versions together. A rule version is identified
by the ``mysql_query_rules`` row of ``runtime_checksums_values`` (``version`` increments on
every LOAD, ``epoch`` says when it was created, ``checksum`` identifies its contents) plus
the generation of the admin connection the rows were read over:

* rule text and hits counted together come from one confirmed version only;
* a reload, an admin connection rebuild, a counter that moved backwards, or a version that
  could not be confirmed opens a breakpoint - the interval is never counted as zero and no
  rate crosses the boundary;
* loading identical rule text again still creates a new version, and deleted rules or
  reused rule ids never rewrite the evidence of an older version.

The tracker is fed one observation per poll while the Query Rules panel is visible. Each
observation also carries everything a replay needs to render the sample offline, so playback
never reads the live proxy to reconstruct history.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from dolphie.DataTypes import DatabaseRow
from dolphie.Modules.Functions import coerce_int, coerce_str

# A sample cannot be linked to the one before it. ``first`` is the baseline observation.
BREAKPOINT_FIRST = "first"
BREAKPOINT_RELOAD = "reload"
BREAKPOINT_RECONNECT = "reconnect"
BREAKPOINT_UNCONFIRMED = "unconfirmed"
BREAKPOINT_ROLLBACK = "rollback"

REPLAY_PAYLOAD_VERSION = 1


@dataclass(frozen=True)
class RuleVersionIdentity:
    """The effective version of a query-rules set.

    Identity never derives from rule text: a no-op LOAD increments ``runtime_version`` and
    must not merge back into the previous version. ``connection_generation`` discriminates
    two reads of the same runtime set over different admin connections: the set is the same
    version, but its counters cannot be followed across the rebuild.
    """

    runtime_version: int
    epoch: int
    checksum: str
    connection_generation: int

    @property
    def runtime_key(self) -> tuple[int, int, str]:
        return (self.runtime_version, self.epoch, self.checksum)


@dataclass
class RuleSample:
    """One poll's view of the query rules and what can be attributed to its version."""

    identity: RuleVersionIdentity | None
    rules: list[DatabaseRow]
    # Hits delta per rule_id against the previous sample of the same confirmed identity.
    # None means the rate for this interval is not attributable (baseline or breakpoint).
    deltas: dict[str, int | None]
    interval: float
    confirmed: bool
    supported: bool
    breakpoint: str | None
    # Session-relative sequence of the identity, assigned the first time it is observed.
    version_seq: int | None
    # What the boundary came from, for breakpoint messages (e.g. previous version/gen).
    breakpoint_detail: dict[str, Any] = field(default_factory=dict)
    # Rule ids whose counters moved backwards within an otherwise continuous version.
    rolled_back_rules: list[str] = field(default_factory=list)
    # monotonic() of the observation; used to space live samples, never serialized
    observed_at: float = 0.0

    def rate_for(self, rule_id: str) -> int | None:
        """Hits per second attributable to this rule for the interval, or None at a boundary."""
        delta = self.deltas.get(rule_id)
        if delta is None or self.interval <= 0:
            return None
        return round(delta / self.interval)

    def to_replay(self) -> dict[str, Any]:
        """Serialize the sample into a replay row payload."""
        identity = None
        if self.identity is not None:
            identity = {
                "runtime_version": self.identity.runtime_version,
                "epoch": self.identity.epoch,
                "checksum": self.identity.checksum,
                "connection_generation": self.identity.connection_generation,
            }
        return {
            "payload_version": REPLAY_PAYLOAD_VERSION,
            "confirmed": self.confirmed,
            "supported": self.supported,
            "interval": self.interval,
            "breakpoint": self.breakpoint,
            "breakpoint_detail": self.breakpoint_detail,
            "rolled_back_rules": self.rolled_back_rules,
            "version_seq": self.version_seq,
            "identity": identity,
            "rules": self.rules,
            "deltas": self.deltas,
        }

    @classmethod
    def from_replay(cls, payload: object) -> RuleSample | None:
        """Rebuild a sample from a replay row payload.

        None means the recording has no query-rule data for that frame.
        """
        if not isinstance(payload, dict):
            return None

        identity_data = payload.get("identity")
        identity: RuleVersionIdentity | None = None
        if isinstance(identity_data, dict):
            checksum = coerce_str(identity_data.get("checksum"))
            if identity_data.get("runtime_version") is not None and checksum:
                identity = RuleVersionIdentity(
                    runtime_version=coerce_int(identity_data.get("runtime_version")),
                    epoch=coerce_int(identity_data.get("epoch")),
                    checksum=checksum,
                    connection_generation=coerce_int(identity_data.get("connection_generation")),
                )

        raw_rules = payload.get("rules")
        rules = [dict(row) for row in raw_rules if isinstance(row, dict)] if isinstance(raw_rules, list) else []

        deltas: dict[str, int | None] = {}
        raw_deltas = payload.get("deltas")
        if isinstance(raw_deltas, dict):
            for rule_id, value in raw_deltas.items():
                deltas[coerce_str(rule_id)] = coerce_int(value) if value is not None else None

        raw_rolled_back = payload.get("rolled_back_rules")
        rolled_back = [coerce_str(rule_id) for rule_id in raw_rolled_back] if isinstance(raw_rolled_back, list) else []
        detail = payload.get("breakpoint_detail")
        version_seq = payload.get("version_seq")

        return cls(
            identity=identity,
            rules=rules,
            deltas=deltas,
            interval=float(payload["interval"]) if payload.get("interval") is not None else 0.0,
            confirmed=bool(payload.get("confirmed", False)),
            supported=bool(payload.get("supported", False)),
            breakpoint=coerce_str(payload["breakpoint"]) if payload.get("breakpoint") else None,
            version_seq=coerce_int(version_seq) if version_seq is not None else None,
            breakpoint_detail=dict(detail) if isinstance(detail, dict) else {},
            rolled_back_rules=rolled_back,
        )


def rule_id_of(row: DatabaseRow) -> str:
    return coerce_str(row.get("rule_id"))


def hits_of(row: DatabaseRow) -> int:
    return coerce_int(row.get("hits"))


class QueryRuleTracker:
    """Builds version-scoped :class:`RuleSample` observations from live polls."""

    def __init__(self) -> None:
        # First observation order of every confirmed runtime set ((version, epoch, checksum)
        # -> v-sequence). The admin connection generation is intentionally not part of this
        # key: reconnecting to an unchanged runtime set is the same effective rule version.
        self._identity_seq: dict[tuple[int, int, str], int] = {}
        # The rule text first observed for each runtime set, retained even after it disappears
        self.evidence: dict[tuple[int, int, str], list[DatabaseRow]] = {}
        self._last: RuleSample | None = None

    @property
    def last_sample(self) -> RuleSample | None:
        return self._last

    def record(
        self,
        rules: list[DatabaseRow],
        *,
        runtime_version: int | None,
        epoch: int | None,
        checksum: str | None,
        connection_generation: int,
        confirmed: bool,
        supported: bool = True,
        observed_at: float | None = None,
    ) -> RuleSample:
        """Record one poll and attribute its hit deltas to a confirmed rule version.

        Returns the sample describing this poll, including any breakpoint against the
        previous one.
        """
        moment = time.monotonic() if observed_at is None else observed_at
        interval = moment - self._last.observed_at if self._last is not None else 0.0

        identity: RuleVersionIdentity | None = None
        if confirmed and runtime_version is not None and epoch is not None and checksum:
            identity = RuleVersionIdentity(
                runtime_version=runtime_version,
                epoch=epoch,
                checksum=checksum,
                connection_generation=connection_generation,
            )

        current_hits = {rule_id_of(row): hits_of(row) for row in rules if rule_id_of(row)}

        breakpoint: str | None = None
        detail: dict[str, Any] = {}
        deltas: dict[str, int | None] = {}
        rolled_back: list[str] = []

        runtime_key = identity.runtime_key if identity is not None else None
        if runtime_key is not None and runtime_key not in self._identity_seq:
            self._identity_seq[runtime_key] = len(self._identity_seq) + 1
            self.evidence[runtime_key] = [dict(row) for row in rules]

        if self._last is None:
            breakpoint = BREAKPOINT_FIRST
            deltas = dict.fromkeys(current_hits)
        elif identity is None or not confirmed or self._last.identity is None:
            # A read that could not be pinned to one version, or a resume after one, breaks
            # the series: the interval may straddle a reload or a counter reset
            breakpoint = BREAKPOINT_UNCONFIRMED
            deltas = dict.fromkeys(current_hits)
        elif identity != self._last.identity:
            previous = self._last.identity
            detail = {
                "from_seq": self._last.version_seq,
                "from_runtime_version": previous.runtime_version,
                "from_connection_generation": previous.connection_generation,
            }
            if (
                identity.runtime_version == previous.runtime_version
                and identity.epoch == previous.epoch
                and identity.checksum == previous.checksum
            ):
                breakpoint = BREAKPOINT_RECONNECT
            else:
                breakpoint = BREAKPOINT_RELOAD
            deltas = dict.fromkeys(current_hits)
        else:
            last_sample = self._last
            for rule_id, hits in current_hits.items():
                # Recover the previous sample's raw hits from its stored rules
                previous_rule = next((row for row in last_sample.rules if rule_id_of(row) == rule_id), None)
                if previous_rule is None:
                    # First time this id is seen within the version - baseline only
                    deltas[rule_id] = None
                elif hits < hits_of(previous_rule):
                    # A counter reset that was not announced by a new version: break this
                    # rule's interval instead of treating it as a negative or zero rate
                    deltas[rule_id] = None
                    rolled_back.append(rule_id)
                else:
                    deltas[rule_id] = hits - hits_of(previous_rule)
            if rolled_back:
                breakpoint = BREAKPOINT_ROLLBACK

        sample = RuleSample(
            identity=identity,
            rules=rules,
            deltas=deltas,
            interval=max(interval, 0.0),
            confirmed=confirmed,
            supported=supported,
            breakpoint=breakpoint,
            version_seq=self._identity_seq.get(runtime_key) if runtime_key is not None else None,
            breakpoint_detail=detail,
            rolled_back_rules=rolled_back,
            observed_at=moment,
        )
        self._last = sample
        return sample


def version_status_markup(sample: RuleSample) -> str:
    """One-line description of the version a sample belongs to."""
    identity = sample.identity
    if identity is None or not sample.confirmed:
        return (
            "[$dark_yellow]Rules version unconfirmed for this sample[/$dark_yellow] "
            "[$dark_gray](version metadata missing or a reload happened while reading)[/$dark_gray]"
        )

    checksum = identity.checksum
    return (
        f"[$label]Rules version[/$label] [$highlight]v{sample.version_seq}[/$highlight] "
        f"[$dark_gray](runtime #[$highlight]{identity.runtime_version}[/$highlight], epoch {identity.epoch}, "
        f"checksum {checksum}, admin conn #{identity.connection_generation})[/$dark_gray]"
    )


def breakpoint_markup(sample: RuleSample) -> str | None:
    """The boundary message for a sample, or None when its interval sits inside one version."""
    identity = sample.identity
    detail = sample.breakpoint_detail

    if sample.breakpoint == BREAKPOINT_RELOAD and identity is not None:
        return (
            f"[$dark_yellow]Breakpoint: query rules were reloaded between samples "
            f"(v{detail.get('from_seq')} → v{sample.version_seq}); LOAD restarts the hit counters, "
            "so hits/s is not calculated across the reload[/$dark_yellow]"
        )
    if sample.breakpoint == BREAKPOINT_RECONNECT and identity is not None:
        return (
            f"[$dark_yellow]Breakpoint: the admin connection was rebuilt "
            f"(conn #{detail.get('from_connection_generation')} → #{identity.connection_generation}); "
            "the hit counters cannot be followed across it, so hits/s is not calculated[/$dark_yellow]"
        )
    if sample.breakpoint == BREAKPOINT_UNCONFIRMED:
        return (
            "[$dark_yellow]Breakpoint: the rule version for this interval could not be confirmed "
            "(a reload may have happened mid-sample); hits/s is not calculated[/$dark_yellow]"
        )
    if sample.breakpoint == BREAKPOINT_ROLLBACK:
        rules = ", ".join(sample.rolled_back_rules) or "?"
        return (
            f"[$dark_yellow]Breakpoint: the hits counter moved backwards for rule(s) [$highlight]{rules}"
            "[/$highlight]; their rate is not estimated for this sample[/$dark_yellow]"
        )
    return None
