from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, RadioButton, RadioSet, Static


class GtidCompareModal(ModalScreen[str]):
    """Picks the target replica replay whose marked segment is scanned for the barrier.

    The active replay tab is the master side and its current frame is the barrier; this
    modal only chooses which other open MySQL replay recording is scanned and displays
    the first/last frames the scan will use.
    """

    CSS = """
        GtidCompareModal {
            & > Vertical {
                background: #131626;
                border: tall #384673;
                width: 100;
                height: auto;
                padding: 1 2;

                & > Label {
                    width: 100%;
                    content-align: center middle;
                    padding-bottom: 1;
                }

                & > Static {
                    width: 100%;
                    height: auto;
                    padding-bottom: 1;
                }

                & RadioSet {
                    width: 100%;
                    height: auto;
                    background: #131626;
                    border: none;
                }

                & RadioButton {
                    width: 100%;
                    background: #131626;
                }

                & Horizontal {
                    width: 100%;
                    height: auto;
                    align: center middle;
                    padding-top: 1;
                }
            }

            & #error_response {
                color: #fe5c5c;
                width: 100%;
                height: auto;
                content-align: center middle;
            }
        }
    """
    BINDINGS = [
        Binding("escape", "app.pop_screen", "", show=False),
    ]

    def __init__(self, master_description: str, candidates: list[tuple[str, str]]):
        """Args:
        master_description: Rendered line describing the master frame used as the barrier.
        candidates: (tab id, one-line description) pairs for the target replay tabs.
        """
        super().__init__()
        self.master_description = master_description
        self.candidates = candidates

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("[b]GTID transaction progress comparison[/b]")
            yield Static(self.master_description)
            yield Static(
                "[$dark_gray]Mark the target's first/last segment frames with [b],[/b] and [b].[/b] on its "
                "tab first. Without marks, the whole recording is scanned.[/$dark_gray]"
            )
            with RadioSet(id="target_radios"):
                for tab_id, description in self.candidates:
                    yield RadioButton(description, id=tab_id)
            yield Static(id="error_response")
            with Horizontal():
                yield Button("Compare", id="submit", variant="primary")
                yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        self.query_one("#error_response", Static).display = False
        if self.candidates:
            # Preselect the first target; RadioSet.pressed_index itself is read-only.
            self.query_one("#target_radios RadioButton", RadioButton).toggle()

    @on(Button.Pressed, "#cancel")
    def _cancel(self) -> None:
        self.app.pop_screen()

    @on(Button.Pressed, "#submit")
    def _submit(self) -> None:
        radios = self.query_one("#target_radios", RadioSet)
        pressed = radios.pressed_button
        if pressed is None or pressed.id is None:
            error = self.query_one("#error_response", Static)
            error.display = True
            error.update("Select a target replay recording")
            return

        self.dismiss(pressed.id)
