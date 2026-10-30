from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static

from dolphie.Modules.Functions import escape_markup
from dolphie.Modules.ReplicationRouting import ChannelFilters
from dolphie.Widgets.AutoComplete import AutoComplete, DropdownItem


class RoutingAuditModal(ModalScreen):
    """Explicitly choose the replication channels and each source's actual base tables.

    Tables are typed (with autocomplete from the tables the replica account can read) and
    toggled in with Enter, so the selection stays explicit even when information_schema hides
    objects the account cannot see.
    """

    CSS = """
        RoutingAuditModal {
            & > VerticalScroll {
                background: #131626;
                border: tall #384673;
                width: 90;
                max-height: 90%;
                padding: 1 2;

                & > * {
                    width: 100%;
                    height: auto;
                }
            }

            & .channel_header {
                padding-top: 1;
                text-style: bold;
            }

            & .channel_filters {
                color: #8b93a7;
                padding-bottom: 0;
            }

            & .selected_tables {
                height: auto;
                max-height: 10;
                padding-bottom: 0;
            }

            & Input {
                width: 80;
            }

            & #error_response {
                color: #fe5c5c;
                height: auto;
                padding-top: 1;
            }

            & .button_container {
                height: auto;
                padding-top: 1;
            }
        }
    """

    BINDINGS = [
        Binding("escape", "app.pop_screen", "", show=False),
    ]

    def __init__(
        self,
        channels: list[tuple[str, ChannelFilters]],
        base_table_candidates: list[tuple[str, str]],
        lower_case_table_names: int | None,
    ):
        super().__init__()
        self.channels = channels
        self.candidate_items = [DropdownItem(f"{db}.{table}") for db, table in base_table_candidates]
        self.lower_case_table_names = lower_case_table_names
        # channel name -> selected explicit source table refs in insertion order
        self.selected_tables: dict[str, list[str]] = {channel: [] for channel, _filters in channels}

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Label(
                "[b]Multi-Source Replication Table Routing Audit[/b]\n"
                "[$dark_gray]Pick each channel's actual source base tables. Enter adds the typed "
                "schema.table; Enter on an already added table removes it. Only channels with at "
                "least one table are audited.[/$dark_gray]"
            )
            lctn_display = (
                str(self.lower_case_table_names)
                if self.lower_case_table_names is not None
                else "[$yellow]unreadable - case-dependent results will be undeterminable[/$yellow]"
            )
            yield Label(f"[$label]lower_case_table_names:[/$label] {lctn_display}")

            for index, (channel, filters) in enumerate(self.channels):
                yield Static(
                    self._channel_title(channel, filters), id=f"channel_title_{index}", classes="channel_header"
                )
                yield Static(self._summarize_filters(filters), classes="channel_filters")
                yield Static(
                    "[$dark_gray]No tables selected yet[/$dark_gray]", id=f"selected_{index}", classes="selected_tables"
                )
                table_input = Input(id=f"table_input_{index}", placeholder="Source base table as schema.table")
                yield table_input
                yield AutoComplete(
                    table_input,
                    id=f"table_complete_{index}",
                    candidates=self.candidate_items,
                    prevent_default_enter=False,
                )

            yield Static(id="error_response")

            with Horizontal(classes="button_container"):
                yield Button("Submit", id="submit", variant="primary")
                yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        self.query_one("#error_response", Static).display = False
        if self.channels:
            self.query_one("#table_input_0", Input).focus()

    @staticmethod
    def parse_table_ref(value: str) -> tuple[str, str]:
        """Validate a ``schema.table`` reference; raise ValueError otherwise."""
        cleaned = value.strip().replace("`", "")
        parts = [part.strip() for part in cleaned.split(".")]
        if len(parts) != 2 or not all(parts):
            raise ValueError("Tables must be given as schema.table with exactly one period")
        return parts[0], parts[1]

    def _channel_index(self, input_id: str) -> int:
        return int(input_id.rsplit("_", 1)[1])

    def _refresh_selected(self, index: int, channel: str) -> None:
        selected = self.selected_tables[channel]
        widget = self.query_one(f"#selected_{index}", Static)
        if selected:
            widget.update("\n".join(f"[$highlight]+[/$highlight] {escape_markup(ref)}" for ref in selected))
        else:
            widget.update("[$dark_gray]No tables selected yet[/$dark_gray]")

    @staticmethod
    def _channel_title(channel: str, filters: ChannelFilters) -> str:
        status = " [$red](configuration unreadable)[/$red]" if filters.blocked else ""
        channel_name = escape_markup(channel) or "[$dark_gray]default channel[/$dark_gray]"
        return f"Channel [$highlight]{channel_name}[/$highlight]{status}"

    @staticmethod
    def _summarize_filters(filters: ChannelFilters) -> str:
        if filters.blocked:
            return "\n".join(f"[$yellow]! {escape_markup(issue)}[/$yellow]" for issue in filters.issues)

        def _names(rules: list) -> list[str]:
            return [rule.raw for rule in rules]

        lines: list[str] = []
        if filters.rewrite_db:
            lines.append(f"rewrite-db: {', '.join(rule.raw for rule in filters.rewrite_db)}")
        for label, rules in (
            ("do-db", _names(filters.do_db)),
            ("ignore-db", _names(filters.ignore_db)),
            ("do-table", _names(filters.do_table)),
            ("ignore-table", _names(filters.ignore_table)),
            ("wild-do-table", _names(filters.wild_do_table)),
            ("wild-ignore-table", _names(filters.wild_ignore_table)),
        ):
            if rules:
                lines.append(f"{label}: {', '.join(rules)}")
        if not lines:
            lines.append("no replication filters (all tables are written)")
        return escape_markup("\n".join(lines))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if not event.input.id or not event.input.id.startswith("table_input_"):
            return

        index = self._channel_index(event.input.id)
        channel, _filters = self.channels[index]
        try:
            db, table = self.parse_table_ref(event.value)
        except ValueError as error:
            self.update_error_response(str(error))
            return

        ref = f"{db}.{table}"
        selected = self.selected_tables[channel]
        if ref in selected:
            selected.remove(ref)
        else:
            selected.append(ref)
        self._refresh_selected(index, channel)
        event.input.value = ""
        self.query_one("#error_response", Static).display = False

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id != "submit":
            self.app.pop_screen()
            return

        # Guard against silently dropping a table that was typed but never added with Enter.
        pending = [
            f"{self.channels[index][0]}: {table_input.value.strip()}"
            for index in range(len(self.channels))
            if (table_input := self.query_one(f"#table_input_{index}", Input)).value.strip()
        ]
        if pending:
            self.update_error_response("Press Enter to add the typed table(s) first: " + ", ".join(pending))
            return

        selection: dict[str, list[tuple[str, str]]] = {}
        for channel, _filters in self.channels:
            tables = []
            for ref in self.selected_tables[channel]:
                try:
                    tables.append(self.parse_table_ref(ref))
                except ValueError:
                    self.update_error_response(f"Invalid table for channel {channel}: {ref}")
                    return
            if tables:
                selection[channel] = tables

        if not selection:
            self.update_error_response("Select at least one source base table on a channel to run the audit")
            return

        self.dismiss(selection)

    def update_error_response(self, message: str) -> None:
        error_response = self.query_one("#error_response", Static)
        error_response.update(message)
        error_response.display = True
