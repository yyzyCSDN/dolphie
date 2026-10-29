from __future__ import annotations

from typing import TYPE_CHECKING, cast

from textual import on
from textual.app import ComposeResult
from textual.containers import Center, Horizontal, Vertical
from textual.widgets import Button, Static

from dolphie.Modules.Functions import escape_markup
from dolphie.Modules.GtidReplaySync import BarrierMatchResult, BarrierStatus
from dolphie.Widgets.DolphieScreen import DolphieScreen, ScreenContext

if TYPE_CHECKING:
    from dolphie.App import DolphieApp

_STATUS_STYLE = {
    BarrierStatus.matched: "[$b_green]MATCHED[/$b_green]",
    BarrierStatus.not_reached: "[$b_yellow]NOT REACHED[/$b_yellow]",
    BarrierStatus.indeterminate: "[$b_red]CANNOT DETERMINE[/$b_red]",
}


def _fold_set(value: str | None) -> str:
    """Renders a GTID set with one entry per line but keeps it selectable as text."""
    if not value:
        return "[$dark_gray]n/a[/$dark_gray]"
    return escape_markup(value.replace(",", ",\n"))


def _frame_line(role: str, replay_id: int | None, timestamp: str | None, filename: str | None) -> str:
    if replay_id is None:
        return f"[$label]{role} frame[/$label]  [$dark_gray]not reached[/$dark_gray]"
    return (
        f"[$label]{role} frame[/$label]  #[$b_light_blue]{replay_id}[/$b_light_blue]"
        f"  [$light_blue]{escape_markup(timestamp or '')}[/$light_blue] UTC"
        f"  [$dark_gray]{escape_markup(filename or '')}[/$dark_gray]"
    )


class GtidCompareResultScreen(DolphieScreen):
    """Shows the barrier scan evidence and jumps to either side's matched frame."""

    CSS = """
        GtidCompareResultScreen {
            & Center {
                padding: 1;

                & > Vertical {
                    padding-left: 1;
                    padding-right: 1;
                    background: #101626;
                    border: tall #1d253e;
                    width: auto;
                    max-width: 160;
                    height: auto;

                    & > Static {
                        width: auto;
                        height: auto;
                        padding-bottom: 1;
                    }

                    & Horizontal {
                        width: 100%;
                        height: auto;
                        align: center middle;
                        padding-top: 1;
                    }
                }
            }
        }
    """

    def __init__(
        self,
        context: ScreenContext,
        result: BarrierMatchResult,
        master_tab_id: str,
        target_tab_id: str,
        master_filename: str,
        target_filename: str,
    ):
        super().__init__(context)
        self.result = result
        self.master_tab_id = master_tab_id
        self.target_tab_id = target_tab_id
        self.master_filename = master_filename
        self.target_filename = target_filename

    def _render_body(self) -> str:
        result = self.result
        lines = [
            "[b]GTID transaction progress comparison[/b]",
            f"Status: {_STATUS_STYLE[result.status]}",
        ]
        if result.reason:
            lines.append(f"[$dark_gray]{escape_markup(result.reason)}[/$dark_gray]")
        lines.append("")

        lines.append(_frame_line("Master", result.master_replay_id, result.master_timestamp, self.master_filename))
        if result.master_server_uuid:
            lines.append(f"  [$dark_gray]server_uuid[/$dark_gray] {escape_markup(result.master_server_uuid)}")
        if result.master_gtid_executed:
            lines.append("  [$dark_gray]gtid_executed (barrier)[/$dark_gray]")
            lines.append(_fold_set(result.master_gtid_executed))
        lines.append("")

        if result.segment_start_id is not None:
            lines.append(
                f"[$label]Target segment[/$label]  #[$b_light_blue]{result.segment_start_id}[/$b_light_blue] "
                f"[$dark_gray]->[/$dark_gray] #[$b_light_blue]{result.segment_end_id}[/$b_light_blue]"
                f"  [$dark_gray]({result.frames_scanned} frame(s) validated)[/$dark_gray]"
            )
        lines.append(_frame_line("Target", result.target_replay_id, result.target_timestamp, self.target_filename))
        if result.target_server_uuid:
            lines.append(f"  [$dark_gray]server_uuid[/$dark_gray] {escape_markup(result.target_server_uuid)}")
        if result.target_gtid_executed:
            lines.append("  [$dark_gray]gtid_executed[/$dark_gray]")
            lines.append(_fold_set(result.target_gtid_executed))

        if result.matched and result.raw_delay_seconds is not None:
            lines.append("")
            lines.append(
                "[$label]Raw wall-clock delta[/$label] "
                f"[$b_highlight]{result.raw_delay_seconds:+.0f} s[/$b_highlight] "
                "[$dark_gray](unadjusted; may include clock skew between the hosts - matching used GTID sets, "
                "not times)[/$dark_gray]"
            )

        lines.append("")
        lines.append(
            "[$dark_gray]Only observed set containment is proven. Row data equality, wall-clock proximity, "
            "Retrieved_Gtid_Set, per-channel rows, and resets missed between samples are not used or "
            "inferred.[/$dark_gray]"
        )
        return "\n".join(lines)

    def compose_content(self) -> ComposeResult:
        # Children are passed as constructor objects (like CommandScreen) rather than yielded
        # inside `with` blocks: this generator is driven through DolphieScreen's `yield from`,
        # and mount context managers opened in a delegated generator exit before their yielded
        # children attach.
        buttons: list[Button] = []
        if self.result.matched:
            buttons.append(Button("View master frame", id="goto_master"))
            buttons.append(Button("View target frame", id="goto_target"))
        buttons.append(Button("Close", id="close"))

        yield Center(
            Vertical(
                Static(self._render_body(), markup=True, id="gtid_result_body"),
                Horizontal(*buttons),
            )
        )

    @on(Button.Pressed, "#goto_master")
    def _goto_master(self) -> None:
        self.dismiss(None)
        cast("DolphieApp", self.app).goto_gtid_compare_tab(self.master_tab_id)

    @on(Button.Pressed, "#goto_target")
    def _goto_target(self) -> None:
        self.dismiss(None)
        cast("DolphieApp", self.app).goto_gtid_compare_tab(self.target_tab_id)

    @on(Button.Pressed, "#close")
    def _close(self) -> None:
        self.dismiss(None)
