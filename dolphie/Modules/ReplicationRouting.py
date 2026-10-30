"""Read-only audit of how ROW events for explicit source tables route through MySQL multi-source replication.

Scope (by design):
    * MySQL/MariaDB configuration that is currently readable from the server: the effective
      per-channel filter set reported by SHOW REPLICA STATUS / SHOW ALL SLAVES STATUS,
      ``@@GLOBAL.lower_case_table_names``, and information_schema base tables.
    * ROW-format row events for an explicit set of source base tables the user chooses,
      assumed to have already been delivered to the chosen channel.

Out of scope:
    * Event-level rules keyed on the event's origin (such as IGNORE_SERVER_IDS). The audit
      starts after such gates, at the table-map database/table that the applier evaluates.
    * STATEMENT events (their do-db/ignore-db semantics depend on the default database),
      tables that do not exist yet, and row-content/primary-key conflicts.
    * Any modification of replication configuration.

Filter priority implemented here is the server's ROW order: rewrite-db is applied first,
then database-level rules (the affected database, not the default database), then the
table-level cascade do-table -> ignore-table -> wild-do-table -> wild-ignore-table, with the
"any allow-list existed but nothing matched" catch-all. See MySQL Reference Manual
"Evaluation of Database-Level/Table-Level Replication Options".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from rich.align import Align
from rich.console import Group, RenderableType

from dolphie.Modules.Functions import coerce_int, coerce_str, escape_markup
from dolphie.Modules.Queries import MySQLQueries
from dolphie.Modules.Theme import ThemedTable as Table
from dolphie.Modules.Theme import themed_text

if TYPE_CHECKING:
    from dolphie.DataTypes import DatabaseRow
    from dolphie.Modules.MySQL import Database

# SHOW REPLICA STATUS / SHOW ALL SLAVES STATUS columns that carry the effective filters.
_DO_DB_FIELD = "Replicate_Do_DB"
_IGNORE_DB_FIELD = "Replicate_Ignore_DB"
_DO_TABLE_FIELD = "Replicate_Do_Table"
_IGNORE_TABLE_FIELD = "Replicate_Ignore_Table"
_WILD_DO_TABLE_FIELD = "Replicate_Wild_Do_Table"
_WILD_IGNORE_TABLE_FIELD = "Replicate_Wild_Ignore_Table"
_REWRITE_DB_FIELD = "Replicate_Rewrite_DB"

_FILTER_FIELDS = (
    _DO_DB_FIELD,
    _IGNORE_DB_FIELD,
    _DO_TABLE_FIELD,
    _IGNORE_TABLE_FIELD,
    _WILD_DO_TABLE_FIELD,
    _WILD_IGNORE_TABLE_FIELD,
    _REWRITE_DB_FIELD,
)
_TABLE_FIELDS = (_DO_TABLE_FIELD, _IGNORE_TABLE_FIELD)
_WILD_FIELDS = (_WILD_DO_TABLE_FIELD, _WILD_IGNORE_TABLE_FIELD)


class Verdict(str, Enum):
    """What happens to a row event that reaches the applier on its channel."""

    write = "WRITE"
    excluded = "EXCLUDED"
    undeterminable = "UNDETERMINABLE"


@dataclass(frozen=True)
class RewriteRule:
    """One ``from_db->to_db`` mapping. Mappings are first-match and never chained."""

    from_db: str
    to_db: str
    raw: str
    ordinal: int


@dataclass(frozen=True)
class ExactRule:
    """A do-db/ignore-db entry (table is None) or a do-table/ignore-table entry."""

    db: str
    table: str | None
    raw: str
    ordinal: int


@dataclass(frozen=True)
class WildRule:
    """A LIKE-style wild rule. ``db_pattern is None`` means the pattern matches any database."""

    db_pattern: str | None
    table_pattern: str
    raw: str
    ordinal: int


@dataclass
class ChannelFilters:
    """The effective ROW filter set for one channel, plus anything that makes it unreadable."""

    channel: str
    available: bool = True
    do_db: list[ExactRule] = field(default_factory=list)
    ignore_db: list[ExactRule] = field(default_factory=list)
    do_table: list[ExactRule] = field(default_factory=list)
    ignore_table: list[ExactRule] = field(default_factory=list)
    wild_do_table: list[WildRule] = field(default_factory=list)
    wild_ignore_table: list[WildRule] = field(default_factory=list)
    rewrite_db: list[RewriteRule] = field(default_factory=list)
    # A non-fatal parsing/unsupported-construct problem: every conclusion for this channel is
    # then marked undeterminable rather than guessing around the unreadable rule.
    issues: list[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return not self.available or bool(self.issues)


@dataclass(frozen=True)
class TraceStep:
    """One evaluated stage of the server's decision cascade."""

    stage: str
    detail: str


@dataclass(frozen=True)
class _ScenarioResult:
    verdict: Verdict
    target_db: str | None
    target_table: str | None
    trace: list[TraceStep]
    # Compact rule tokens, e.g. ("rewrite:src_a->mrg", "wild-do:mrg.ord%"), for collision rows.
    path_tokens: list[str]


@dataclass
class TableRoute:
    channel: str
    source_db: str
    source_table: str
    verdict: Verdict = Verdict.undeterminable
    target_db: str | None = None
    target_table: str | None = None
    # Targets the event might write to when the verdict is undeterminable (empty when even the
    # target cannot be computed, e.g. the channel's rules are unreadable).
    possible_targets: set[tuple[str, str]] = field(default_factory=set)
    # One trace per name-case scenario ("effective" when lower_case_table_names is known).
    traces: list[tuple[str, _ScenarioResult]] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    @property
    def target(self) -> tuple[str, str] | None:
        if self.target_db is not None and self.target_table is not None:
            return (self.target_db, self.target_table)
        return None


@dataclass(frozen=True)
class CollisionSide:
    channel: str
    source_db: str
    source_table: str
    definite: bool
    target_db: str
    target_table: str
    path_tokens: tuple[str, ...]


@dataclass
class Collision:
    target_db: str
    target_table: str
    # Confirmed: every side deterministically writes the target. Undeterminable: at least one
    # side's write depends on information that could not be read.
    confirmed: bool
    sides: list[CollisionSide] = field(default_factory=list)


@dataclass
class AuditNotice:
    channel: str | None
    message: str


@dataclass
class RoutingAudit:
    lower_case_table_names: int | None
    routes: list[TableRoute] = field(default_factory=list)
    collisions: list[Collision] = field(default_factory=list)
    notices: list[AuditNotice] = field(default_factory=list)


@dataclass
class RoutingAuditConfig:
    """Read-only inputs gathered before the user picks the explicit table sets."""

    filters_by_channel: dict[str, ChannelFilters]
    lower_case_table_names: int | None
    base_tables: list[tuple[str, str]]
    notices: list[str] = field(default_factory=list)


def _split_entries(value: str | None) -> list[str]:
    # SHOW REPLICA STATUS joins multiple values with commas and no surrounding whitespace.
    return [entry.strip() for entry in coerce_str(value).split(",") if entry.strip()]


def _parse_db_rules(channel: str, raw_value: str | None, field_name: str) -> tuple[list[ExactRule], list[str]]:
    rules: list[ExactRule] = []
    issues: list[str] = []
    for ordinal, entry in enumerate(_split_entries(raw_value)):
        if "." in entry:
            # Database filters hold bare database names; a dot means the SHOW output cannot be
            # split unambiguously (quoted identifiers may contain commas/dots).
            issues.append(f"Channel {channel!r}: cannot interpret {field_name} entry {entry!r} as a database name")
            continue
        rules.append(ExactRule(db=entry, table=None, raw=entry, ordinal=ordinal))
    return rules, issues


def _parse_table_rules(channel: str, raw_value: str | None, field_name: str) -> tuple[list[ExactRule], list[str]]:
    rules: list[ExactRule] = []
    issues: list[str] = []
    for ordinal, entry in enumerate(_split_entries(raw_value)):
        parts = entry.split(".")
        if len(parts) != 2 or not all(parts):
            # The server joins a comma-separated list with no quoting, so an entry without
            # exactly one "db.table" split is one we cannot map back to a concrete table.
            issues.append(f"Channel {channel!r}: cannot interpret {field_name} entry {entry!r}; expected schema.table")
            continue
        rules.append(ExactRule(db=parts[0], table=parts[1], raw=entry, ordinal=ordinal))
    return rules, issues


def _parse_wild_rules(channel: str, raw_value: str | None, field_name: str) -> tuple[list[WildRule], list[str]]:
    rules: list[WildRule] = []
    issues: list[str] = []
    for ordinal, entry in enumerate(_split_entries(raw_value)):
        # The server splits at the first dot; a dotless pattern matches a table in any database.
        db_pattern, separator, table_pattern = entry.partition(".")
        if not separator or not table_pattern:
            if not separator:
                rules.append(WildRule(db_pattern=None, table_pattern=entry, raw=entry, ordinal=ordinal))
            else:
                issues.append(f"Channel {channel!r}: {field_name} pattern {entry!r} has an empty table name")
            continue
        rules.append(WildRule(db_pattern=db_pattern, table_pattern=table_pattern, raw=entry, ordinal=ordinal))
    return rules, issues


def _parse_rewrite_rules(channel: str, raw_value: str | None) -> tuple[list[RewriteRule], list[str]]:
    rules: list[RewriteRule] = []
    issues: list[str] = []
    for ordinal, entry in enumerate(_split_entries(raw_value)):
        from_db, arrow, to_db = entry.partition("->")
        if not arrow or not from_db or not to_db or "->" in to_db:
            issues.append(
                f"Channel {channel!r}: cannot interpret {_REWRITE_DB_FIELD} entry {entry!r}; expected from->to"
            )
            continue
        rules.append(RewriteRule(from_db=from_db, to_db=to_db, raw=entry, ordinal=ordinal))
    return rules, issues


def parse_channel_filters(channel: str, row: DatabaseRow | None) -> ChannelFilters:
    """Build the effective filter set from one SHOW REPLICA STATUS row.

    ``row is None`` (privilege failure or a status row we could not read) leaves the channel
    available=False; every route through it is then undeterminable.
    """
    if row is None:
        return ChannelFilters(
            channel=channel,
            available=False,
            issues=[
                (
                    f"Channel {channel!r}: replication status could not be read (missing REPLICATION CLIENT "
                    "privilege or the channel status row is unavailable)"
                )
            ],
        )

    filters = ChannelFilters(channel=channel)
    issues: list[str] = filters.issues

    for field_name in _FILTER_FIELDS:
        if field_name not in row:
            issues.append(
                f"Channel {channel!r}: {field_name} is not reported by this server/version; "
                "this rule set cannot be evaluated"
            )

    for field_name, target in ((_DO_DB_FIELD, "do_db"), (_IGNORE_DB_FIELD, "ignore_db")):
        rules, field_issues = _parse_db_rules(channel, coerce_str(row.get(field_name)), field_name)
        getattr(filters, target).extend(rules)
        issues.extend(field_issues)

    for field_name, target in ((_DO_TABLE_FIELD, "do_table"), (_IGNORE_TABLE_FIELD, "ignore_table")):
        rules, field_issues = _parse_table_rules(channel, coerce_str(row.get(field_name)), field_name)
        getattr(filters, target).extend(rules)
        issues.extend(field_issues)

    for field_name, target in (
        (_WILD_DO_TABLE_FIELD, "wild_do_table"),
        (_WILD_IGNORE_TABLE_FIELD, "wild_ignore_table"),
    ):
        rules, field_issues = _parse_wild_rules(channel, coerce_str(row.get(field_name)), field_name)
        getattr(filters, target).extend(rules)
        issues.extend(field_issues)

    rewrite_rules, field_issues = _parse_rewrite_rules(channel, coerce_str(row.get(_REWRITE_DB_FIELD)))
    filters.rewrite_db.extend(rewrite_rules)
    issues.extend(field_issues)

    return filters


def like_match(pattern: str, text: str, *, case_sensitive: bool) -> bool:
    r"""Match a replication wildcard pattern with LIKE semantics.

    ``%`` matches any run of characters, ``_`` exactly one, and a backslash escapes the next
    character (so ``\%``, ``\_`` and ``\\`` are literals). A dangling trailing backslash can
    never match, mirroring the server's LIKE comparison.
    """
    if not case_sensitive:
        pattern = pattern.lower()
        text = text.lower()

    text_index = 0
    pattern_index = 0
    star_pattern_index = -1
    star_text_index = 0

    while text_index < len(text):
        if pattern_index < len(pattern) and pattern[pattern_index] == "\\":
            # Escaping beyond the pattern end is an impossible match on the server.
            if pattern_index + 1 >= len(pattern):
                return False
            if pattern[pattern_index + 1] == text[text_index]:
                pattern_index += 2
                text_index += 1
                continue
            if star_pattern_index != -1:
                pattern_index = star_pattern_index + 1
                star_text_index += 1
                text_index = star_text_index
                continue
            return False
        if pattern_index < len(pattern) and pattern[pattern_index] == "%":
            star_pattern_index = pattern_index
            star_text_index = text_index
            pattern_index += 1
            continue
        if pattern_index < len(pattern) and (
            pattern[pattern_index] == "_" or pattern[pattern_index] == text[text_index]
        ):
            pattern_index += 1
            text_index += 1
            continue
        if star_pattern_index != -1:
            pattern_index = star_pattern_index + 1
            star_text_index += 1
            text_index = star_text_index
            continue
        return False

    # Only trailing % wildcards may match the empty remainder.
    while pattern_index < len(pattern) and pattern[pattern_index] == "%":
        pattern_index += 1
    return pattern_index == len(pattern)


def _names_equal(left: str, right: str, *, case_sensitive: bool) -> bool:
    if case_sensitive:
        return left == right
    return left.lower() == right.lower()


def _exact_rule_matches(rule: ExactRule, db: str, table: str, *, case_sensitive: bool) -> bool:
    if not _names_equal(rule.db, db, case_sensitive=case_sensitive):
        return False
    return rule.table is None or _names_equal(rule.table, table, case_sensitive=case_sensitive)


def _wild_rule_matches(rule: WildRule, db: str, table: str, *, case_sensitive: bool) -> bool:
    if rule.db_pattern is not None and not like_match(rule.db_pattern, db, case_sensitive=case_sensitive):
        return False
    return like_match(rule.table_pattern, table, case_sensitive=case_sensitive)


def _first_rewrite(filters: ChannelFilters, db: str, *, case_sensitive: bool) -> RewriteRule | None:
    # Rewrites are first-match; the server never chains a second mapping.
    for rule in filters.rewrite_db:
        if _names_equal(rule.from_db, db, case_sensitive=case_sensitive):
            return rule
    return None


def _evaluate_scenario(
    filters: ChannelFilters, source_db: str, source_table: str, *, case_sensitive: bool
) -> _ScenarioResult:
    """Evaluate the full ROW cascade for one fixed name-case behaviour."""
    trace: list[TraceStep] = []
    path_tokens: list[str] = []

    # Stage 0: database rewriting precedes every --replicate-* rule.
    rewrite = _first_rewrite(filters, source_db, case_sensitive=case_sensitive)
    if rewrite is not None:
        target_db = rewrite.to_db
        path_tokens.append(f"rewrite:{rewrite.raw}")
        trace.append(
            TraceStep(
                "rewrite-db",
                f"Matched rewrite rule #{rewrite.ordinal + 1} {rewrite.raw!r}: database "
                f"{source_db!r} becomes {target_db!r} before any filter is tested",
            )
        )
    else:
        target_db = source_db
        if filters.rewrite_db:
            trace.append(TraceStep("rewrite-db", f"No rewrite rule matches {source_db!r}; database is unchanged"))

    # Stage 1 (ROW: the database the row changes, not the default database).
    if filters.do_db:
        matched = next(
            (rule for rule in filters.do_db if _names_equal(rule.db, target_db, case_sensitive=case_sensitive)),
            None,
        )
        if matched is None:
            trace.append(
                TraceStep(
                    "do-db",
                    f"replicate-do-db allowlist is set and {target_db!r} is not on it; excluded before "
                    "table rules are checked",
                )
            )
            return _ScenarioResult(Verdict.excluded, None, None, trace, path_tokens)
        path_tokens.append(f"do-db:{matched.raw}")
        trace.append(TraceStep("do-db", f"Matched replicate-do-db {matched.raw!r}; continue to table rules"))
    elif filters.ignore_db:
        matched = next(
            (rule for rule in filters.ignore_db if _names_equal(rule.db, target_db, case_sensitive=case_sensitive)),
            None,
        )
        if matched is not None:
            path_tokens.append(f"ignore-db:{matched.raw}")
            trace.append(TraceStep("ignore-db", f"Matched replicate-ignore-db {matched.raw!r}; excluded"))
            return _ScenarioResult(Verdict.excluded, None, None, trace, path_tokens)
        trace.append(TraceStep("ignore-db", f"No replicate-ignore-db rule matches {target_db!r}; continue"))

    # Stage 2: table-level cascade in server priority order.
    def matches(rule: ExactRule | WildRule) -> bool:
        if isinstance(rule, WildRule):
            return _wild_rule_matches(rule, target_db, source_table, case_sensitive=case_sensitive)
        return _exact_rule_matches(rule, target_db, source_table, case_sensitive=case_sensitive)

    table_checks: tuple[tuple[str, list[ExactRule] | list[WildRule], Verdict, bool], ...] = (
        ("do-table", filters.do_table, Verdict.write, True),
        ("ignore-table", filters.ignore_table, Verdict.excluded, False),
        ("wild-do-table", filters.wild_do_table, Verdict.write, True),
        ("wild-ignore-table", filters.wild_ignore_table, Verdict.excluded, False),
    )
    for stage, rules, verdict_on_match, is_write in table_checks:
        if not rules:
            continue
        matched = next((rule for rule in rules if matches(rule)), None)
        if matched is not None:
            path_tokens.append(f"{stage}:{matched.raw}")
            action = "written" if is_write else "excluded"
            trace.append(TraceStep(stage, f"Matched replicate-{stage} {matched.raw!r}; event is {action}"))
            target = (target_db, source_table) if is_write else (None, None)
            return _ScenarioResult(verdict_on_match, target[0], target[1], trace, path_tokens)
        trace.append(TraceStep(stage, f"No replicate-{stage} rule matches {target_db}.{source_table}"))

    # Catch-all: an allowlist existed but nothing on it matched.
    if filters.do_table or filters.wild_do_table:
        trace.append(
            TraceStep(
                "decision",
                "replicate-do-table/replicate-wild-do-table allowlists exist but none matched; excluded",
            )
        )
        return _ScenarioResult(Verdict.excluded, None, None, trace, path_tokens)

    path_tokens.append("default-accept")
    trace.append(TraceStep("decision", "No rule excluded the event; written to the post-rewrite table"))
    return _ScenarioResult(Verdict.write, target_db, source_table, trace, path_tokens)


def _scenario_label(case_sensitive: bool, *, known: bool) -> str:
    mode = "case-sensitive name comparison (lower_case_table_names=0)"
    if not case_sensitive:
        mode = "case-insensitive name comparison (lower_case_table_names=1 or 2)"
    return mode if known else f"If server uses {mode}"


def route_table(
    filters: ChannelFilters,
    source_db: str,
    source_table: str,
    lower_case_table_names: int | None,
) -> TableRoute:
    """Decide one explicit source table's ROW event through one channel."""
    route = TableRoute(channel=filters.channel, source_db=source_db, source_table=source_table)

    if filters.blocked:
        route.verdict = Verdict.undeterminable
        route.reasons.extend(filters.issues)
        return route

    if lower_case_table_names is None:
        scenarios = ((True,), (False,))
        known_case = False
    else:
        scenarios = ((lower_case_table_names == 0,),)
        known_case = True

    results: list[tuple[str, _ScenarioResult]] = []
    for (case_sensitive,) in scenarios:
        label = _scenario_label(case_sensitive, known=known_case)
        results.append((label, _evaluate_scenario(filters, source_db, source_table, case_sensitive=case_sensitive)))

    route.traces = results
    verdicts = {result.verdict for _label, result in results}
    targets: set[tuple[str, str]] = set()
    for _label, result in results:
        if result.verdict is Verdict.write and result.target_db is not None and result.target_table is not None:
            targets.add((result.target_db, result.target_table))

    sole_verdict = next(iter(verdicts)) if len(verdicts) == 1 else None
    if sole_verdict is Verdict.excluded:
        route.verdict = Verdict.excluded
    elif sole_verdict is Verdict.write and len(targets) == 1:
        target_db, target_table = next(iter(targets))
        route.verdict = Verdict.write
        route.target_db = target_db
        route.target_table = target_table
    else:
        # Mixed verdicts, or a write target that differs depending on name case: nothing can
        # be asserted as definite, but the candidate write targets stay visible for collisions.
        route.verdict = Verdict.undeterminable
        route.possible_targets = targets
        case_sensitive_result = results[0][1]
        case_insensitive_result = results[1][1]
        route.reasons.append(
            "lower_case_table_names could not be read, so name case semantics are unknown. "
            f"Case-sensitive path: {case_sensitive_result.verdict.value}"
            + (
                f" -> {case_sensitive_result.target_db}.{case_sensitive_result.target_table}"
                if case_sensitive_result.target_db
                else ""
            )
            + f"; case-insensitive path: {case_insensitive_result.verdict.value}"
            + (
                f" -> {case_insensitive_result.target_db}.{case_insensitive_result.target_table}"
                if case_insensitive_result.target_db
                else ""
            )
        )

    return route


def run_routing_audit(
    channel_inputs: list[tuple[ChannelFilters, list[tuple[str, str]]]],
    lower_case_table_names: int | None,
) -> RoutingAudit:
    """Audit explicit base-table sets per channel and detect cross-channel target collisions."""
    audit = RoutingAudit(lower_case_table_names=lower_case_table_names)

    if lower_case_table_names is None:
        audit.notices.append(
            AuditNotice(
                None,
                "@@GLOBAL.lower_case_table_names could not be read: conclusions that depend on the "
                "case sensitivity of database/table names are marked undeterminable",
            )
        )

    routes: list[TableRoute] = []
    for filters, tables in channel_inputs:
        if filters.blocked:
            audit.notices.extend(AuditNotice(filters.channel, issue) for issue in filters.issues)
        deduped_tables = sorted(set(tables))
        for source_db, source_table in deduped_tables:
            routes.append(route_table(filters, source_db, source_table, lower_case_table_names))
    audit.routes = routes

    audit.collisions = _find_collisions(routes, lower_case_table_names)

    # Routes whose target could not even be computed cannot be cleared against any other
    # channel; call those out so an unreadable channel is never read as "no collision".
    for route in routes:
        if route.verdict is Verdict.undeterminable and not route.possible_targets:
            audit.notices.append(
                AuditNotice(
                    route.channel,
                    f"Route of {route.source_db}.{route.source_table} is undeterminable before a target "
                    "table could be computed; collisions involving it cannot be ruled out: " + "; ".join(route.reasons),
                )
            )

    return audit


def _case_insensitive_equal(left: tuple[str, str], right: tuple[str, str]) -> bool:
    return left[0].lower() == right[0].lower() and left[1].lower() == right[1].lower()


def _find_collisions(routes: list[TableRoute], lower_case_table_names: int | None) -> list[Collision]:
    # target -> claims. Definite claims come from WRITE routes, possible claims from
    # undeterminable routes with candidate targets.
    claims: dict[tuple[str, str], list[CollisionSide]] = {}
    for route in routes:
        primary_tokens = tuple(route.traces[0][1].path_tokens) if route.traces else ()
        if route.verdict is Verdict.write and route.target is not None:
            claims.setdefault(route.target, []).append(
                CollisionSide(
                    channel=route.channel,
                    source_db=route.source_db,
                    source_table=route.source_table,
                    definite=True,
                    target_db=route.target[0],
                    target_table=route.target[1],
                    path_tokens=primary_tokens,
                )
            )
        elif route.verdict is Verdict.undeterminable:
            for target_db, target_table in sorted(route.possible_targets):
                claims.setdefault((target_db, target_table), []).append(
                    CollisionSide(
                        channel=route.channel,
                        source_db=route.source_db,
                        source_table=route.source_table,
                        definite=False,
                        target_db=target_db,
                        target_table=target_table,
                        path_tokens=primary_tokens,
                    )
                )

    # Cluster exact targets that name the same table under the server's case semantics. With
    # lower_case_table_names 1/2 different-case targets are one table; when the setting cannot
    # be read such a merge is itself uncertain and can only yield an undeterminable collision.
    targets = list(claims)
    parent = {target: target for target in targets}

    def find(target: tuple[str, str]) -> tuple[str, str]:
        while parent[target] != target:
            parent[target] = parent[parent[target]]
            target = parent[target]
        return target

    def union(left: tuple[str, str], right: tuple[str, str]) -> None:
        parent[find(right)] = find(left)

    case_uncertain_clusters: set[tuple[str, str]] = set()
    for index, left in enumerate(targets):
        for right in targets[index + 1 :]:
            if left == right or not _case_insensitive_equal(left, right):
                continue
            if lower_case_table_names in (1, 2):
                union(left, right)
            elif lower_case_table_names is None:
                union(left, right)
                case_uncertain_clusters.add(find(left))

    clusters: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for target in targets:
        clusters.setdefault(find(target), []).append(target)

    collisions: list[Collision] = []
    for root, members in clusters.items():
        sides = [side for target in members for side in claims[target]]
        distinct_channels = {side.channel for side in sides}
        if len(distinct_channels) < 2:
            # Same source/channel converging onto one table is not a multi-source collision.
            continue

        case_uncertain = find(root) in case_uncertain_clusters
        definite_channels = {side.channel for side in sides if side.definite}
        confirmed = (
            not case_uncertain
            and len(definite_channels) >= 2
            and len(sides) == len([side for side in sides if side.definite])
        )
        representative = min(members)
        collisions.append(
            Collision(
                target_db=representative[0],
                target_table=representative[1],
                confirmed=confirmed,
                sides=sorted(sides, key=lambda side: (not side.definite, side.channel, side.source_db)),
            )
        )

    return sorted(
        collisions,
        key=lambda collision: (not collision.confirmed, collision.target_db, collision.target_table),
    )


def load_routing_audit_config(
    connection: Database | None,
    status_rows: list[DatabaseRow],
) -> RoutingAuditConfig:
    """Gather the read-only inputs the audit needs from a live replica connection.

    The effective per-channel filters come from the already-polled SHOW REPLICA STATUS rows;
    this function additionally reads lower_case_table_names and the base-table candidates the
    user picks from. It never runs a write statement.
    """
    notices: list[str] = []
    filters_by_channel: dict[str, ChannelFilters] = {}
    for row in status_rows:
        channel = coerce_str(row.get("Channel_Name"))
        filters_by_channel[channel] = parse_channel_filters(channel, row)

    lower_case_table_names: int | None = None
    if connection is not None:
        value = connection.fetch_value_from_field(
            "SELECT @@GLOBAL.lower_case_table_names AS lower_case_table_names",
            "lower_case_table_names",
            ignore_error=True,
        )
        if value is not None:
            lower_case_table_names = coerce_int(value)
        else:
            notices.append(
                "@@GLOBAL.lower_case_table_names could not be read; name-case-dependent conclusions "
                "will be marked undeterminable"
            )

        connection.execute(MySQLQueries.base_tables, ignore_error=True)
        if connection.last_execute_successful:
            base_tables = sorted(
                (coerce_str(row.get("table_schema")), coerce_str(row.get("table_name")))
                for row in connection.fetchall()
            )
        else:
            base_tables = []
            notices.append(
                "information_schema base tables could not be listed (missing privileges): table selection "
                "accepts typed schema.table names, but the candidate list is incomplete"
            )
    else:
        base_tables = []

    return RoutingAuditConfig(
        filters_by_channel=filters_by_channel,
        lower_case_table_names=lower_case_table_names,
        base_tables=base_tables,
        notices=notices,
    )


def _format_table_ref(db: str, table: str) -> str:
    return f"[$dark_gray]{escape_markup(db)}[/$dark_gray].{escape_markup(table)}"


def _verdict_style(verdict: Verdict) -> str:
    if verdict is Verdict.write:
        return "$green"
    if verdict is Verdict.excluded:
        return "$dark_gray"
    return "$yellow"


def _render_trace(route: TableRoute) -> str:
    blocks: list[str] = []
    for label, result in route.traces:
        steps = "\n".join(
            f"[$label]{escape_markup(step.stage)}[/$label] {escape_markup(step.detail)}" for step in result.trace
        )
        if len(route.traces) > 1:
            blocks.append(f"[$b_light_blue]{escape_markup(label)}[/$b_light_blue]\n{steps}")
        else:
            blocks.append(steps)
    reason_text = ""
    if route.reasons:
        reason_text = "\n" + "\n".join(f"[$yellow]! {escape_markup(reason)}[/$yellow]" for reason in route.reasons)
    return "\n".join(blocks) + reason_text


def _route_target_cell(route: TableRoute) -> str:
    if route.verdict is Verdict.write and route.target is not None:
        return _format_table_ref(route.target[0], route.target[1])
    if route.verdict is Verdict.undeterminable and route.possible_targets:
        return "[$yellow]possibly:[/$yellow] " + ", ".join(
            _format_table_ref(db, table) for db, table in sorted(route.possible_targets)
        )
    return "[$dark_gray]—[/$dark_gray]"


def build_routing_audit_renderable(audit: RoutingAudit) -> Group:
    """Render the audit as Rich content for a CommandScreen (read-only display)."""
    sections: list[RenderableType] = []

    lctn = audit.lower_case_table_names
    lctn_display = (
        "[$yellow]unreadable (case-dependent results are undeterminable)[/$yellow]" if lctn is None else str(lctn)
    )
    sections.append(
        Align.center(
            themed_text(
                "[$b_light_blue]Multi-Source Replication Table Routing Audit[/$b_light_blue]\n"
                f"[$label]lower_case_table_names:[/$label] {lctn_display}    "
                f"[$label]tables audited:[/$label] {len(audit.routes)}    "
                f"[$label]potential collisions:[/$label] {len(audit.collisions)}\n"
            )
        )
    )

    routes_by_channel: dict[str, list[TableRoute]] = {}
    for route in audit.routes:
        routes_by_channel.setdefault(route.channel, []).append(route)

    for channel, channel_routes in routes_by_channel.items():
        channel_table = Table(box=None, title=f"Channel [$highlight]{escape_markup(channel)}[/$highlight]")
        channel_table.add_column("Source Table", no_wrap=True)
        channel_table.add_column("Outcome", no_wrap=True)
        channel_table.add_column("Target Table", no_wrap=True)
        channel_table.add_column("Decision Rule Path", overflow="fold")

        for route in sorted(channel_routes, key=lambda item: (item.source_db, item.source_table)):
            color = _verdict_style(route.verdict)
            channel_table.add_row(
                _format_table_ref(route.source_db, route.source_table),
                f"[{color}]{route.verdict.value}[/{color}]",
                _route_target_cell(route),
                _render_trace(route),
            )
        sections.append(channel_table)

    if audit.collisions:
        collision_table = Table(box=None, title="[$b_light_blue]Potential Cross-Channel Target Collisions")
        collision_table.add_column("Target Table", no_wrap=True)
        collision_table.add_column("Status", no_wrap=True)
        collision_table.add_column("Sources and Rule Paths", overflow="fold")

        for collision in audit.collisions:
            status = "[$red]CONFIRMED[/$red]" if collision.confirmed else "[$yellow]UNDETERMINABLE[/$yellow]"
            sides_text = "\n".join(
                (
                    f"{('[$green]write[/$green]' if side.definite else '[$yellow]possible write[/$yellow]')} "
                    f"[$highlight]{escape_markup(side.channel)}[/$highlight]: "
                    f"{_format_table_ref(side.source_db, side.source_table)}"
                    + (
                        f" writes {_format_table_ref(side.target_db, side.target_table)}"
                        if (side.target_db, side.target_table) != (collision.target_db, collision.target_table)
                        else ""
                    )
                    + f" [$dark_gray]({' -> '.join(escape_markup(token) for token in side.path_tokens)})[/$dark_gray]"
                )
                for side in collision.sides
            )
            collision_table.add_row(_format_table_ref(collision.target_db, collision.target_table), status, sides_text)
        sections.append(collision_table)

    if audit.notices:
        notice_table = Table(box=None, show_header=False)
        notice_table.add_column()
        for notice in audit.notices:
            scope = f"[[$highlight]{escape_markup(notice.channel)}[/$highlight]] " if notice.channel is not None else ""
            notice_table.add_row(f"[$yellow]! {scope}{escape_markup(notice.message)}[/$yellow]")
        sections.append(notice_table)

    sections.append(
        Align.center(
            themed_text(
                "[$dark_gray]Scope: explicit base tables assumed delivered as ROW events to the selected "
                "channels. Rewrite-db is applied before do/ignore database and table filters using the "
                "server's ROW priority. Event-origin rules (e.g. IGNORE_SERVER_IDS), STATEMENT events, "
                "future tables and row-content conflicts are not assessed. No replication configuration "
                "was modified.[/$dark_gray]"
            )
        )
    )

    return Group(*sections)
