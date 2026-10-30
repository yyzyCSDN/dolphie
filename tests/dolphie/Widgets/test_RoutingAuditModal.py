import pytest
from textual.app import App
from textual.widgets import Button, Input

from dolphie.Modules.ReplicationRouting import parse_channel_filters
from dolphie.Widgets.RoutingAuditModal import RoutingAuditModal


def channels():
    return [
        ("channel_a", parse_channel_filters("channel_a", {"Channel_Name": "channel_a"})),
        ("channel_b", parse_channel_filters("channel_b", {"Channel_Name": "channel_b"})),
    ]


async def drive_modal(
    adds: list[tuple[int, str]],
    *,
    typed: list[tuple[int, str]] | None = None,
    submit: bool = True,
    candidates=None,
):
    """Open the audit modal, toggle in ``adds`` (Enter pressed), optionally leave ``typed`` text
    unsubmitted, and submit. Returns (dismissed, error)."""
    app = App()
    dismissed = []

    async with app.run_test(size=(120, 40)) as pilot:
        modal = RoutingAuditModal(channels(), candidates or [("src_a", "orders"), ("src_b", "orders")], 0)
        app.push_screen(modal, dismissed.append)
        await pilot.pause()

        for index, ref in adds:
            table_input = modal.query_one(f"#table_input_{index}", Input)
            table_input.value = ref
            table_input.post_message(Input.Submitted(table_input, ref))
            await pilot.pause()

        for index, ref in typed or []:
            table_input = modal.query_one(f"#table_input_{index}", Input)
            table_input.focus()
            table_input.value = ref
            await pilot.pause()

        if submit:
            modal.query_one("#submit", Button).press()
            await pilot.pause()

        error_widget = modal.query("#error_response")
        error = None
        if error_widget and error_widget.first().display:
            error = str(error_widget.first().render())

    return dismissed[0] if dismissed else None, error


async def test_modal_rejects_an_empty_selection():
    dismissed, error = await drive_modal([])

    assert dismissed is None
    assert error is not None
    assert "at least one source base table" in error.lower()


async def test_modal_returns_the_explicit_tables_grouped_by_channel():
    dismissed, error = await drive_modal([(0, "src_a.orders"), (1, "src_b.orders"), (1, "src_b.secret")])

    assert error is None
    assert dismissed == {
        "channel_a": [("src_a", "orders")],
        "channel_b": [("src_b", "orders"), ("src_b", "secret")],
    }


async def test_modal_toggles_a_table_back_off():
    dismissed, error = await drive_modal([(0, "src_a.orders"), (0, "src_a.orders")])

    assert dismissed is None
    assert error is not None
    assert "at least one source base table" in error.lower()


async def test_modal_rejects_a_ref_without_schema_table_shape():
    # The Enter toggle is rejected before submit, so the bad value never becomes a selection.
    _dismissed, error = await drive_modal([(0, "not_a_table")], submit=False)

    assert error is not None
    assert "schema.table" in error


async def test_unsubmitted_text_blocks_submit_instead_of_being_dropped():
    dismissed, error = await drive_modal([(0, "src_a.orders")], typed=[(1, "src_b.leftover")])

    assert dismissed is None
    assert error is not None
    assert "Press Enter" in error
    assert "src_b.leftover" in error


async def test_modal_marks_unreadable_channel_configuration():
    app = App()
    async with app.run_test(size=(120, 40)) as pilot:
        broken = parse_channel_filters("dark", None)
        modal = RoutingAuditModal([("dark", broken)], [], None)
        app.push_screen(modal)
        await pilot.pause()

        title = str(modal.query_one("#channel_title_0").render())
        assert "dark" in title
        assert "unreadable" in title


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("db.table", ("db", "table")), (" `db`.`table` ", ("db", "table"))],
)
def test_parse_table_ref(raw: str, expected: tuple[str, str]):
    assert RoutingAuditModal.parse_table_ref(raw) == expected
