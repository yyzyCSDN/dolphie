"""Replication panel against real multi-node topologies: async replicas, Group Replication, Galera,
multi-source channels, and an InnoDB ClusterSet."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, cast

import pytest
from rich.console import Console
from rich.console import Group as RichGroup
from textual.widgets import Input, Static

from dolphie.DataTypes import ConnectionSource
from dolphie.Widgets.CommandScreen import CommandScreen
from dolphie.Widgets.RoutingAuditModal import RoutingAuditModal
from tests.integration.harness import DolphieHarness, make_config, query, run_dolphie
from tests.integration.servers import Server, Topology

mariadb = pytest.mark.parametrize("topology", ["mariadb"], indirect=True)
gr = pytest.mark.parametrize("topology", ["gr"], indirect=True)
galera = pytest.mark.parametrize("topology", ["galera"], indirect=True)
multi_source = pytest.mark.parametrize("topology", ["multi-source"], indirect=True)
clusterset = pytest.mark.parametrize("topology", ["clusterset"], indirect=True)


async def open_replication_panel(harness: DolphieHarness) -> None:
    await harness.wait_for_polls(2)
    await harness.press("4")
    assert harness.dolphie.panels.replication.visible, harness.notifications
    assert harness.tab.panel_replication.display


async def assert_replica_channels(harness: DolphieHarness, count: int) -> None:
    await harness.wait_for(lambda: len(harness.dolphie.replication_status) == count, message=f"{count} channels")
    for status in harness.dolphie.replication_status:
        assert status.get("Replica_IO_Running", status.get("Slave_IO_Running")) == "Yes", status
    await open_replication_panel(harness)
    assert harness.tab.replication_container.display


@mariadb
async def test_mariadb_primary_discovers_and_polls_its_replicas(topology: Topology, tmp_path: Path) -> None:
    primary = topology.endpoints["primary"]

    async with run_dolphie(make_config(primary, tmp_path)) as harness:
        await harness.wait_for_polls(2)
        assert harness.dolphie.connection_source_alt == ConnectionSource.mariadb
        await harness.wait_for(lambda: harness.dolphie.replica_manager.discovery_count == 2, message="2 replicas")

        await open_replication_panel(harness)
        # The replicas worker connects to each replica through its report-host and report-port.
        await harness.wait_for(lambda: harness.dolphie.replica_manager.active_count == 2, message="2 replica polls")
        await harness.wait_for(
            lambda: not harness.tab.replicas_loading_indicator.display, message="replica tables to render"
        )
        assert harness.tab.replicas_container.display


@mariadb
async def test_mariadb_replica_shows_its_channel(topology: Topology, tmp_path: Path) -> None:
    replica = topology.endpoints["replica_1"]

    async with run_dolphie(make_config(replica, tmp_path)) as harness:
        await assert_replica_channels(harness, 1)
        assert harness.dolphie.global_variables.get("read_only") == "ON"


@gr
async def test_group_replication_members(topology: Topology, tmp_path: Path) -> None:
    node = topology.endpoints["node1"]

    async with run_dolphie(make_config(node, tmp_path)) as harness:
        await harness.wait_for_polls(2)
        assert harness.dolphie.group_replication
        assert not harness.dolphie.innodb_cluster
        await harness.wait_for(lambda: len(harness.dolphie.group_replication_members) == 3, message="3 members")
        assert {m.get("MEMBER_STATE") for m in harness.dolphie.group_replication_members} == {"ONLINE"}

        await open_replication_panel(harness)
        assert harness.tab.group_replication_container.display


@gr
async def test_async_replica_of_a_group(topology: Topology, tmp_path: Path) -> None:
    replica = topology.endpoints["async_replica"]

    async with run_dolphie(make_config(replica, tmp_path)) as harness:
        await assert_replica_channels(harness, 1)
        assert not harness.dolphie.group_replication


@galera
async def test_galera_cluster_nodes(topology: Topology, tmp_path: Path) -> None:
    node = topology.endpoints["node1"]

    async with run_dolphie(make_config(node, tmp_path)) as harness:
        await harness.wait_for_polls(2)
        assert harness.dolphie.galera_cluster
        await harness.wait_for(
            lambda: harness.dolphie.global_status.get("wsrep_cluster_size") == 3, message="3 galera nodes"
        )
        assert harness.dolphie.global_status.get("wsrep_ready") == "ON"
        await open_replication_panel(harness)


@galera
async def test_async_replica_of_a_galera_cluster(topology: Topology, tmp_path: Path) -> None:
    replica = topology.endpoints["async_replica"]

    async with run_dolphie(make_config(replica, tmp_path)) as harness:
        await assert_replica_channels(harness, 1)
        assert not harness.dolphie.galera_cluster


@multi_source
async def test_multi_source_replica_shows_every_channel(topology: Topology, tmp_path: Path) -> None:
    replica = topology.endpoints["replica"]

    async with run_dolphie(make_config(replica, tmp_path)) as harness:
        await assert_replica_channels(harness, 2)
        channels = {str(status.get("Channel_Name")) for status in harness.dolphie.replication_status}
        assert len(channels) == 2
        assert "" not in channels


async def _wait_for_scalar(server: Server, sql: str, expected: int, *, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = int(query(server, sql)[0]["c"])
        if value == expected:
            return
        await asyncio.sleep(0.5)
    raise AssertionError(f"Timed out waiting for {sql!r} == {expected}")


@multi_source
async def test_multi_source_routing_audit_reports_rewrite_collision_and_exclusion(
    topology: Topology, tmp_path: Path
) -> None:
    primary_a = topology.endpoints["primary_a"]
    primary_b = topology.endpoints["primary_b"]
    replica = topology.endpoints["replica"]

    # Source tables whose ROW events the audit will be asked about.
    query(primary_a, "CREATE DATABASE IF NOT EXISTS audit_a")
    query(primary_a, "CREATE TABLE IF NOT EXISTS audit_a.orders (id INT PRIMARY KEY)")
    query(primary_b, "CREATE DATABASE IF NOT EXISTS audit_b")
    query(primary_b, "CREATE TABLE IF NOT EXISTS audit_b.orders (id INT PRIMARY KEY)")
    query(primary_b, "CREATE TABLE IF NOT EXISTS audit_b.secret (id INT PRIMARY KEY)")

    # DDL replicates before the filters go in, so the source tables exist on the replica.
    await _wait_for_scalar(
        replica,
        "SELECT COUNT(*) AS c FROM information_schema.tables WHERE table_schema='audit_b' AND table_name='secret'",
        1,
    )

    # The rewrite target must pre-exist; replication never creates the destination table.
    query(replica, "CREATE DATABASE IF NOT EXISTS mrg")
    query(replica, "CREATE TABLE IF NOT EXISTS mrg.orders (id INT PRIMARY KEY)")

    apply_filters = [
        "STOP REPLICA SQL_THREAD FOR CHANNEL 'channel_a'",
        "CHANGE REPLICATION FILTER REPLICATE_REWRITE_DB=((audit_a,mrg)) FOR CHANNEL 'channel_a'",
        "START REPLICA FOR CHANNEL 'channel_a'",
        "STOP REPLICA SQL_THREAD FOR CHANNEL 'channel_b'",
        (
            "CHANGE REPLICATION FILTER REPLICATE_REWRITE_DB=((audit_b,mrg)), "
            "REPLICATE_WILD_IGNORE_TABLE=((mrg.secret)) FOR CHANNEL 'channel_b'"
        ),
        "START REPLICA FOR CHANNEL 'channel_b'",
    ]
    reset_filters = [
        "STOP REPLICA SQL_THREAD FOR CHANNEL 'channel_a'",
        "CHANGE REPLICATION FILTER REPLICATE_REWRITE_DB=() FOR CHANNEL 'channel_a'",
        "START REPLICA FOR CHANNEL 'channel_a'",
        "STOP REPLICA SQL_THREAD FOR CHANNEL 'channel_b'",
        ("CHANGE REPLICATION FILTER REPLICATE_REWRITE_DB=(), REPLICATE_WILD_IGNORE_TABLE=() FOR CHANNEL 'channel_b'"),
        "START REPLICA FOR CHANNEL 'channel_b'",
    ]

    try:
        for statement in apply_filters:
            query(replica, statement)

        query(primary_a, "INSERT INTO audit_a.orders VALUES (1) ON DUPLICATE KEY UPDATE id=id")
        query(primary_b, "INSERT INTO audit_b.orders VALUES (1) ON DUPLICATE KEY UPDATE id=id")
        query(primary_b, "INSERT INTO audit_b.secret VALUES (1) ON DUPLICATE KEY UPDATE id=id")

        # Both channels rewrite into mrg.orders; the secret row is filtered out on channel_b.
        await _wait_for_scalar(replica, "SELECT COUNT(*) AS c FROM mrg.orders", 2)
        await asyncio.sleep(2)
        assert int(query(replica, "SELECT COUNT(*) AS c FROM audit_b.secret")[0]["c"]) == 0

        async with run_dolphie(make_config(replica, tmp_path)) as harness:
            await harness.wait_for_polls(2)

            await harness.press("G")
            await harness.wait_for(
                lambda: any(isinstance(screen, RoutingAuditModal) for screen in harness.app.screen_stack),
                message="routing audit modal",
            )
            modal = next(screen for screen in harness.app.screen_stack if isinstance(screen, RoutingAuditModal))

            def channel_index(channel: str) -> int:
                for index in range(2):
                    title_widget = cast(Any, modal.query_one(f"#channel_title_{index}", Static))
                    if channel in str(title_widget.renderable):
                        return index
                raise AssertionError(f"channel {channel} not found in audit modal")

            async def add_table(channel: str, ref: str) -> None:
                index = channel_index(channel)
                table_input = modal.query_one(f"#table_input_{index}", Input)
                table_input.focus()
                table_input.value = ref
                table_input.cursor_position = len(ref)
                table_input.post_message(Input.Submitted(table_input, ref))
                # A no-op key in the single-line input pumps the posted Submitted message.
                await harness.press("down")

            await add_table("channel_a", "audit_a.orders")
            await add_table("channel_b", "audit_b.orders")
            await add_table("channel_b", "audit_b.secret")

            await harness.click_button("#submit")
            await harness.wait_for(
                lambda: isinstance(harness.app.screen, CommandScreen),
                message="routing audit report screen",
            )

            report = next(
                renderable
                for widget in harness.app.screen.query(Static)
                if isinstance((renderable := cast(Any, widget).renderable), RichGroup)
            )
            console = Console(force_terminal=False, no_color=True, width=250)
            with console.capture() as capture:
                console.print(report)
            text = capture.get()

            assert "CONFIRMED" in text
            assert "EXCLUDED" in text
            assert "mrg.orders" in text
            assert "audit_a.orders" in text
            assert "audit_b.orders" in text
            assert "rewrite:audit_a->mrg" in text
            assert "wild-ignore-table:mrg.secret" in text
            assert "UNDETERMINABLE" not in text
    finally:
        for statement in reset_filters:
            query(replica, statement)


@clusterset
async def test_clusterset_primary_and_replica_cluster(topology: Topology, tmp_path: Path) -> None:
    primary = topology.endpoints["primary_node1"]
    replica_primary = topology.endpoints["replica_node4"]

    async with run_dolphie(make_config(primary, tmp_path)) as harness:
        await harness.wait_for_polls(2)
        assert harness.dolphie.innodb_cluster
        assert not harness.dolphie.group_replication
        await harness.wait_for(lambda: len(harness.dolphie.group_replication_members) == 3, message="3 members")
        await open_replication_panel(harness)
        assert harness.tab.group_replication_container.display

    async with run_dolphie(make_config(replica_primary, tmp_path)) as harness:
        await harness.wait_for_polls(2)
        assert harness.dolphie.innodb_cluster
        # The replica cluster's primary pulls from the primary cluster over the clusterset channel.
        await harness.wait_for(
            lambda: any(
                status.get("Channel_Name") == "clusterset_replication" for status in harness.dolphie.replication_status
            ),
            message="clusterset_replication channel",
        )
        await open_replication_panel(harness)
