from __future__ import annotations

import concurrent.futures
import json
import math
import socket
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import cast

import mlx.core as mx
import pytest

from exo.worker.engines.mlx.win32_ring import (
    WinRingGroup,
    install_win32_ring_patch,
    uninstall_win32_ring_patch,
)


def _allocate_loopback_ports(count: int) -> list[str]:
    sockets: list[socket.socket] = []
    hosts: list[str] = []
    try:
        for _ in range(count):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            port = cast("tuple[str, int]", sock.getsockname())[1]
            sockets.append(sock)
            hosts.append(f"127.0.0.1:{port}")
    finally:
        for sock in sockets:
            sock.close()
    return hosts


def _run_ring_ranks[T](
    world_size: int,
    fn: Callable[[int, WinRingGroup], T],
) -> list[T]:
    hosts = _allocate_loopback_ports(world_size)

    def worker(rank: int) -> T:
        group: WinRingGroup | None = None
        try:
            group = WinRingGroup.from_hosts(hosts, rank, connect_timeout=15.0)
            return fn(rank, group)
        finally:
            if group is not None:
                group.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=world_size) as pool:
        future_to_rank = {pool.submit(worker, rank): rank for rank in range(world_size)}
        ordered: list[T | None] = [None] * world_size
        for future in concurrent.futures.as_completed(future_to_rank, timeout=20.0):
            ordered[future_to_rank[future]] = future.result()

    return [cast("T", result) for result in ordered]


def test_singleton_group() -> None:
    group = WinRingGroup.singleton()
    assert group.rank() == 0
    assert group.size() == 1
    assert group.split(0).size() == 1

    scalar = mx.array(3.5, dtype=mx.bfloat16)
    assert mx.array_equal(group.all_sum(scalar), scalar)
    gathered_scalar = group.all_gather(scalar)
    assert gathered_scalar.shape == (1,)
    assert mx.array_equal(
        gathered_scalar.view(mx.uint16), scalar.reshape(1).view(mx.uint16)
    )

    vec = mx.array([[1, 2], [3, 4]], dtype=mx.int32)
    assert mx.array_equal(group.all_gather(vec), vec)

    with pytest.raises(RuntimeError):
        _ = group.send(vec, 0)
    with pytest.raises(RuntimeError):
        _ = group.recv((2, 2), mx.int32, 0)


def test_from_env_validation_and_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLX_HOSTFILE", raising=False)
    monkeypatch.delenv("MLX_RANK", raising=False)

    with pytest.raises(RuntimeError):
        _ = WinRingGroup.from_env(strict=True)

    non_strict = WinRingGroup.from_env(strict=False)
    assert non_strict.rank() == 0
    assert non_strict.size() == 1

    with tempfile.TemporaryDirectory() as tmpdir:
        hostfile = Path(tmpdir) / "hosts.json"
        _ = hostfile.write_text(json.dumps([["127.0.0.1:50001"]]), encoding="utf-8")
        monkeypatch.setenv("MLX_HOSTFILE", str(hostfile))
        single = WinRingGroup.from_env(strict=True)
        assert single.rank() == 0
        assert single.size() == 1


def test_non_neighbor_rejected_for_size_four() -> None:
    group = WinRingGroup(rank=0, size=4)
    try:
        arr = mx.array([1.0], dtype=mx.float32)
        with pytest.raises(RuntimeError, match="non-neighbor"):
            _ = group.send(arr, 2)
        with pytest.raises(RuntimeError, match="non-neighbor"):
            _ = group.recv((1,), mx.float32, 2)
    finally:
        group.close()


def test_send_recv_and_recv_like_two_ranks() -> None:
    payload_f32 = mx.array([[1.5, -2.25], [3.75, 4.0]], dtype=mx.float32)
    payload_bf16 = mx.array([1.25, -3.5, 0.5, 64.0], dtype=mx.bfloat16)
    expected_non_contig = mx.array([[0, 3], [1, 4], [2, 5]], dtype=mx.int32)
    scalar_bool = mx.array(True, dtype=mx.bool_)

    def run_rank(rank: int, group: WinRingGroup) -> dict[str, mx.array]:
        if rank == 0:
            sent_f32 = group.send(payload_f32, 1)
            assert sent_f32 is payload_f32
            rx_bf16 = group.recv(payload_bf16.shape, mx.bfloat16, 1)
            # Transposed (non-contiguous) source array must serialize in row-major order.
            non_contig = mx.arange(6, dtype=mx.int32).reshape(2, 3).T
            _ = group.send(non_contig, 1)
            rx_bool = group.recv_like(mx.array(False, dtype=mx.bool_), 1)
            return {"rx_bf16": rx_bf16, "rx_bool": rx_bool}

        rx_f32 = group.recv_like(mx.zeros_like(payload_f32), 0)
        _ = group.send(payload_bf16, 0)
        rx_non_contig = group.recv(expected_non_contig.shape, mx.int32, 0)
        _ = group.send(scalar_bool, 0)
        return {"rx_f32": rx_f32, "rx_non_contig": rx_non_contig}

    outputs = _run_ring_ranks(2, run_rank)
    assert mx.array_equal(outputs[1]["rx_f32"], payload_f32)
    assert mx.array_equal(
        outputs[0]["rx_bf16"].view(mx.uint16),
        payload_bf16.view(mx.uint16),
    )
    assert mx.array_equal(outputs[1]["rx_non_contig"], expected_non_contig)
    assert bool(outputs[0]["rx_bool"].item()) is True
    assert outputs[0]["rx_bool"].shape == ()


def test_send_recv_three_ranks() -> None:
    def run_rank(rank: int, group: WinRingGroup) -> tuple[mx.array, mx.array]:
        right = (rank + 1) % 3
        left = (rank + 2) % 3
        msg_right = mx.array([float(rank) + 0.5, float(rank) * 2.0], dtype=mx.bfloat16)
        msg_left = mx.array([rank + 10], dtype=mx.int32)

        # Rank 0 sends right first to break circle cleanly.
        if rank == 0:
            _ = group.send(msg_right, right)
            from_left = group.recv_like(msg_right, left)
            _ = group.send(msg_left, left)
            from_right = group.recv_like(msg_left, right)
        else:
            from_left = group.recv_like(msg_right, left)
            _ = group.send(msg_right, right)
            from_right = group.recv_like(msg_left, right)
            _ = group.send(msg_left, left)
        return from_left, from_right

    outputs = _run_ring_ranks(3, run_rank)
    for rank in range(3):
        left = (rank + 2) % 3
        right = (rank + 1) % 3
        expected_left = mx.array(
            [float(left) + 0.5, float(left) * 2.0], dtype=mx.bfloat16
        )
        expected_right = mx.array([right + 10], dtype=mx.int32)
        assert mx.array_equal(
            outputs[rank][0].view(mx.uint16), expected_left.view(mx.uint16)
        )
        assert mx.array_equal(outputs[rank][1], expected_right)


@pytest.mark.parametrize("world_size", [2, 3])
def test_all_gather_shapes_dtypes_and_large_buffer(world_size: int) -> None:
    def run_rank(rank: int, group: WinRingGroup) -> tuple[mx.array, mx.array, mx.array]:
        scalar = mx.array(rank + 1, dtype=mx.int32)
        bf16_mat = mx.array(
            [[float(rank) + 0.25, float(rank) - 0.5]], dtype=mx.bfloat16
        )
        # 512x512 float32 = 1 MiB per rank to exceed default socket buffers.
        large = mx.full((512, 512), float(rank + 1), dtype=mx.float32)
        return (
            group.all_gather(scalar),
            group.all_gather(bf16_mat),
            group.all_gather(large),
        )

    outputs = _run_ring_ranks(world_size, run_rank)
    expected_scalars = mx.array(list(range(1, world_size + 1)), dtype=mx.int32)
    expected_bf16 = mx.concatenate(
        [
            mx.array([[float(r) + 0.25, float(r) - 0.5]], dtype=mx.bfloat16)
            for r in range(world_size)
        ],
        axis=0,
    )
    expected_large = mx.concatenate(
        [
            mx.full((512, 512), float(r + 1), dtype=mx.float32)
            for r in range(world_size)
        ],
        axis=0,
    )

    for gathered_scalar, gathered_bf16, gathered_large in outputs:
        assert mx.array_equal(gathered_scalar, expected_scalars)
        assert mx.array_equal(
            gathered_bf16.view(mx.uint16), expected_bf16.view(mx.uint16)
        )
        assert gathered_large.shape == (512 * world_size, 512)
        assert mx.array_equal(gathered_large, expected_large)


@pytest.mark.parametrize("world_size", [2, 3])
def test_all_sum_smaller_than_world_size(world_size: int) -> None:
    def run_rank(
        rank: int, group: WinRingGroup
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
        barrier_val = group.all_sum(mx.array([1.0], dtype=mx.float32))
        one_bf16 = group.all_sum(mx.array([float(rank) + 1.25], dtype=mx.bfloat16))
        one_int32 = group.all_sum(mx.array([rank + 10], dtype=mx.int32))
        any_true = group.all_sum(mx.array(rank == world_size - 1, dtype=mx.bool_))
        all_false = group.all_sum(mx.array(False, dtype=mx.bool_))
        pair_bf16 = group.all_sum(
            mx.array([float(rank) + 0.5, -float(rank)], dtype=mx.bfloat16)
        )
        return barrier_val, one_bf16, one_int32, any_true, all_false, pair_bf16

    outputs = _run_ring_ranks(world_size, run_rank)
    expected_one_bf16 = mx.array(
        [sum(float(r) + 1.25 for r in range(world_size))], dtype=mx.bfloat16
    )
    expected_one_int32 = mx.array(
        [sum(r + 10 for r in range(world_size))], dtype=mx.int32
    )
    expected_bf16 = mx.array(
        [
            sum(float(r) + 0.5 for r in range(world_size)),
            sum(-float(r) for r in range(world_size)),
        ],
        dtype=mx.bfloat16,
    )
    for (
        barrier_val,
        one_bf16,
        one_int32,
        any_true,
        all_false,
        pair_bf16,
    ) in outputs:
        assert barrier_val.shape == (1,)
        barrier_item = barrier_val.item()
        assert isinstance(barrier_item, float)
        assert math.isclose(barrier_item, float(world_size))
        assert one_bf16.shape == (1,)
        assert mx.array_equal(
            one_bf16.view(mx.uint16), expected_one_bf16.view(mx.uint16)
        )
        assert one_int32.shape == (1,)
        assert mx.array_equal(one_int32, expected_one_int32)
        assert any_true.shape == ()
        assert bool(any_true.item()) is True
        assert bool(all_false.item()) is False
        assert mx.array_equal(pair_bf16.view(mx.uint16), expected_bf16.view(mx.uint16))


@pytest.mark.parametrize("world_size", [2, 3])
def test_all_sum_larger_divisible_non_divisible_and_empty_tail(world_size: int) -> None:
    def run_rank(rank: int, group: WinRingGroup) -> tuple[mx.array, mx.array, mx.array]:
        # Divisible 2D bfloat16 tensor
        div_bf16 = mx.full((world_size, 4), float(rank + 1) * 0.5, dtype=mx.bfloat16)
        # Non-divisible length
        non_div_f32 = mx.arange(world_size * 2 + 1, dtype=mx.float32) + float(rank)
        # length=4 on world_size=3 -> seg_size=2 -> tail segment has 0 elements!
        tail_zero = mx.array([rank + 1, rank + 2, rank + 3, rank + 4], dtype=mx.int32)
        return (
            group.all_sum(div_bf16),
            group.all_sum(non_div_f32),
            group.all_sum(tail_zero),
        )

    outputs = _run_ring_ranks(world_size, run_rank)
    expected_div_scalar = sum(float(r + 1) * 0.5 for r in range(world_size))
    expected_div_bf16 = mx.full((world_size, 4), expected_div_scalar, dtype=mx.bfloat16)
    expected_non_div = sum(
        (mx.arange(world_size * 2 + 1, dtype=mx.float32) + float(r))
        for r in range(world_size)
    )
    expected_tail = sum(
        mx.array([r + 1, r + 2, r + 3, r + 4], dtype=mx.int32)
        for r in range(world_size)
    )

    for out_div, out_non_div, out_tail in outputs:
        assert mx.array_equal(
            out_div.view(mx.uint16), expected_div_bf16.view(mx.uint16)
        )
        assert mx.array_equal(out_non_div, expected_non_div)
        assert mx.array_equal(out_tail, expected_tail)


def test_patched_mx_distributed_api(monkeypatch: pytest.MonkeyPatch) -> None:
    install_win32_ring_patch()
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            hostfile = Path(tmpdir) / "hosts.json"
            _ = hostfile.write_text(json.dumps(["127.0.0.1:50099"]), encoding="utf-8")
            monkeypatch.setenv("MLX_HOSTFILE", str(hostfile))
            monkeypatch.setenv("MLX_RANK", "0")

            group = mx.distributed.init(backend="ring", strict=True)
            assert group.rank() == 0
            assert group.size() == 1

            val = mx.array([2.5, -1.5], dtype=mx.bfloat16)
            summed = mx.distributed.all_sum(val, group=group)
            gathered = mx.distributed.all_gather(val, group=group)
            assert mx.array_equal(summed.view(mx.uint16), val.view(mx.uint16))
            assert mx.array_equal(gathered.view(mx.uint16), val.view(mx.uint16))

            # Calling without group uses the cached default WinRingGroup even
            # after the temporary hostfile is deleted.
        assert not hostfile.exists()
        default_sum = mx.distributed.all_sum(mx.array([3.0], dtype=mx.float32))
        default_item = default_sum.item()
        assert isinstance(default_item, float)
        assert math.isclose(default_item, 3.0)

        def run_patched_two_rank(
            rank: int, ring_group: WinRingGroup
        ) -> tuple[mx.array, mx.array, mx.array]:
            dist_group = cast("mx.distributed.Group", cast("object", ring_group))
            bf16_vec = mx.array(
                [float(rank) + 1.0, float(rank) + 2.5], dtype=mx.bfloat16
            )
            if rank == 0:
                _ = mx.distributed.send(bf16_vec, 1, group=dist_group)
                rx = mx.distributed.recv_like(bf16_vec, 1, group=dist_group)
            else:
                rx = mx.distributed.recv(
                    bf16_vec.shape, mx.bfloat16, 0, group=dist_group
                )
                _ = mx.distributed.send(bf16_vec, 0, group=dist_group)
            summed_two = mx.distributed.all_sum(bf16_vec, group=dist_group)
            gathered_two = mx.distributed.all_gather(bf16_vec, group=dist_group)
            return rx, summed_two, gathered_two

        two_rank_out = _run_ring_ranks(2, run_patched_two_rank)
        expected_rx_0 = mx.array([2.0, 3.5], dtype=mx.bfloat16)
        expected_rx_1 = mx.array([1.0, 2.5], dtype=mx.bfloat16)
        expected_sum = mx.array([3.0, 6.0], dtype=mx.bfloat16)
        expected_gather = mx.array([1.0, 2.5, 2.0, 3.5], dtype=mx.bfloat16)

        assert mx.array_equal(
            two_rank_out[0][0].view(mx.uint16), expected_rx_0.view(mx.uint16)
        )
        assert mx.array_equal(
            two_rank_out[1][0].view(mx.uint16), expected_rx_1.view(mx.uint16)
        )
        for _, summed_two, gathered_two in two_rank_out:
            assert mx.array_equal(
                summed_two.view(mx.uint16), expected_sum.view(mx.uint16)
            )
            assert mx.array_equal(
                gathered_two.view(mx.uint16), expected_gather.view(mx.uint16)
            )
    finally:
        uninstall_win32_ring_patch()


def test_concurrent_large_bidirectional_send_does_not_deadlock() -> None:
    payload_floats = 6 * 1024 * 1024  # 24 MiB of float32 (exceeds small TCP buffers)

    def exchange_simultaneously(
        rank: int, group: WinRingGroup
    ) -> tuple[int, float, float]:
        peer = 1 - rank
        outgoing = mx.full((payload_floats,), float(rank + 1), dtype=mx.float32)
        _ = group.send(outgoing, peer)
        incoming = group.recv((payload_floats,), mx.float32, peer)
        mx.eval(incoming)
        first_item = float(incoming[0].item())
        last_item = float(incoming[-1].item())
        return incoming.size, first_item, last_item

    results = _run_ring_ranks(2, exchange_simultaneously)
    assert results[0] == (payload_floats, 2.0, 2.0)
    assert results[1] == (payload_floats, 1.0, 1.0)


def test_ipv6_loopback_ring_if_supported() -> None:
    try:
        probe = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        probe.bind(("::1", 0))
        port0 = cast("tuple[str, int, int, int]", probe.getsockname())[1]
        probe.close()

        probe2 = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        probe2.bind(("::1", 0))
        port1 = cast("tuple[str, int, int, int]", probe2.getsockname())[1]
        probe2.close()
    except OSError:
        pytest.skip("IPv6 loopback unavailable in this environment")

    hosts = [f"[::1]:{port0}", f"[::1]:{port1}"]

    def worker(rank: int) -> mx.array:
        group = WinRingGroup.from_hosts(hosts, rank, connect_timeout=10.0)
        try:
            return group.all_sum(mx.array([rank + 1.0], dtype=mx.float32))
        finally:
            group.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        f0 = pool.submit(worker, 0)
        f1 = pool.submit(worker, 1)
        assert math.isclose(float(f0.result()[0].item()), 3.0)
        assert math.isclose(float(f1.result()[0].item()), 3.0)
