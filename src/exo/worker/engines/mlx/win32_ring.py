from __future__ import annotations

import contextlib
import math
import os
import socket
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import cast, final

import mlx.core as mx
import numpy as np
from loguru import logger
from pydantic import RootModel


class _HostFileEntries(RootModel[list[str] | list[list[str]]]):
    pass


def _parse_host_port(entry: str) -> tuple[str, int]:
    if ":" not in entry:
        raise ValueError(
            f"Invalid MLX_HOSTFILE address (expected host:port): {entry!r}"
        )
    host_part, port_part = entry.rsplit(":", 1)
    if host_part.startswith("[") and host_part.endswith("]"):
        host_part = host_part[1:-1]
    port = int(port_part)
    if not (1 <= port <= 65535):
        raise ValueError(f"Invalid TCP port in MLX_HOSTFILE entry: {entry!r}")
    return host_part, port


def _load_hostfile(hostfile_path: str | Path) -> list[str]:
    parsed = _HostFileEntries.model_validate_json(
        Path(hostfile_path).read_text(encoding="utf-8")
    ).root

    hosts: list[str] = []
    for item in parsed:
        if isinstance(item, str):
            hosts.append(item)
        else:
            if not item:
                raise ValueError(f"Invalid nested MLX_HOSTFILE entry: {item!r}")
            hosts.append(item[0])

    if not hosts:
        raise ValueError("MLX_HOSTFILE contains no hosts")
    return hosts


_RING_SOCKET_BUFFER_BYTES = 16 * 1024 * 1024


def _configure_tcp_socket(sock: socket.socket) -> None:
    sock.settimeout(None)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    for opt in (socket.SO_SNDBUF, socket.SO_RCVBUF):
        with contextlib.suppress(OSError):
            sock.setsockopt(socket.SOL_SOCKET, opt, _RING_SOCKET_BUFFER_BYTES)


def _open_listener(host: str, port: int, *, backlog: int) -> socket.socket:
    if ":" in host:
        family = socket.AF_INET6
    elif host and all(ch.isdigit() or ch == "." for ch in host):
        family = socket.AF_INET
    else:
        family = socket.AF_UNSPEC

    infos = socket.getaddrinfo(
        host or None,
        port,
        family=family,
        type=socket.SOCK_STREAM,
        proto=socket.IPPROTO_TCP,
        flags=socket.AI_PASSIVE,
    )
    last_error: OSError | None = None
    for af, socktype, proto, _, sockaddr in infos:
        listener = socket.socket(af, socktype, proto)
        try:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(sockaddr)
            listener.listen(backlog)
            return listener
        except OSError as exc:
            last_error = exc
            listener.close()
    assert last_error is not None
    raise last_error


def _accept_peer(listener: socket.socket, timeout: float) -> socket.socket:
    listener.settimeout(timeout)
    try:
        peer_sock = listener.accept()[0]
    finally:
        listener.close()
    _configure_tcp_socket(peer_sock)
    return peer_sock


def _connect_peer(host: str, port: int, timeout: float) -> socket.socket:
    deadline = time.monotonic() + timeout
    delay = 0.05
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"Timed out connecting to ring neighbor {host}:{port} after {timeout:.1f}s"
            )
        try:
            attempt_timeout = min(remaining, 2.0)
            sock = socket.create_connection((host, port), timeout=attempt_timeout)
            _configure_tcp_socket(sock)
            return sock
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(min(delay, max(0.01, deadline - time.monotonic())))
            delay = min(delay * 1.5, 0.5)


def _send_exact(sock: socket.socket, data: memoryview) -> None:
    if data.nbytes == 0:
        return
    sock.sendall(data)


def _recv_exact(sock: socket.socket, buf: memoryview) -> None:
    total = buf.nbytes
    if total == 0:
        return
    received = 0
    while received < total:
        chunk = sock.recv_into(buf[received:])
        if chunk == 0:
            raise ConnectionError(
                f"Ring peer closed connection after receiving {received}/{total} bytes"
            )
        received += chunk


def _array_to_bytes(arr: mx.array) -> bytes:
    if arr.size == 0:
        return b""
    flat_u8 = arr.reshape(-1).view(mx.uint8)
    mx.eval(flat_u8)
    return bytes(memoryview(flat_u8))


def _bytes_to_array(
    raw: bytes | bytearray | memoryview,
    dtype: mx.Dtype,
    shape: Sequence[int],
) -> mx.array:
    shape_tuple = tuple(shape)
    numel = math.prod(shape_tuple) if shape_tuple else 1
    if numel == 0:
        return mx.zeros(shape_tuple, dtype=dtype)
    expected_bytes = numel * dtype.size
    mv = memoryview(raw)
    if mv.nbytes != expected_bytes:
        raise ValueError(
            f"Expected {expected_bytes} bytes for shape={shape_tuple} dtype={dtype}, got {mv.nbytes}"
        )
    np_u8 = np.frombuffer(mv, dtype=np.uint8)
    result = mx.reshape(mx.array(np_u8, dtype=mx.uint8).view(dtype), shape_tuple)
    mx.eval(result)
    return result


@final
class WinRingGroup:
    """Pure-Python TCP ring group matching Apple's ``mlx/distributed/ring/ring.cpp`` wire protocol."""

    def __init__(
        self,
        *,
        rank: int,
        size: int,
        sock_left: socket.socket | None = None,
        sock_right: socket.socket | None = None,
    ) -> None:
        self._rank = rank
        self._size = size
        self._sock_left = sock_left
        self._sock_right = sock_right
        self._send_executor: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"win-ring-{rank}")
            if size > 1
            else None
        )
        self._send_futures: list[Future[None]] = []

    @classmethod
    def singleton(cls) -> WinRingGroup:
        return cls(rank=0, size=1)

    @classmethod
    def from_hosts(
        cls,
        hosts: Sequence[str],
        rank: int,
        *,
        connect_timeout: float = 60.0,
    ) -> WinRingGroup:
        size = len(hosts)
        if size == 0:
            raise ValueError("Host list must not be empty")
        if not (0 <= rank < size):
            raise ValueError(f"Rank {rank} out of bounds for size {size}")
        if size == 1:
            return cls.singleton()

        bind_host, bind_port = _parse_host_port(hosts[rank])
        right = (rank + 1) % size
        right_host, right_port = _parse_host_port(hosts[right])

        listener = _open_listener(bind_host, bind_port, backlog=max(size, 4))

        sock_left: socket.socket | None = None
        sock_right: socket.socket | None = None
        try:
            # Match ring.cpp:395 handshake order (rank < right accepts first;
            # the highest rank connects first to break the ring cycle).
            if rank < right:
                sock_left = _accept_peer(listener, connect_timeout)
                sock_right = _connect_peer(right_host, right_port, connect_timeout)
            else:
                sock_right = _connect_peer(right_host, right_port, connect_timeout)
                sock_left = _accept_peer(listener, connect_timeout)
        except Exception:
            for sock in (listener, sock_left, sock_right):
                if sock is not None:
                    sock.close()
            raise

        return cls(rank=rank, size=size, sock_left=sock_left, sock_right=sock_right)

    @classmethod
    def from_env(
        cls,
        *,
        strict: bool = True,
        connect_timeout: float = 60.0,
    ) -> WinRingGroup:
        hostfile = os.environ.get("MLX_HOSTFILE")
        if not hostfile:
            if strict:
                raise RuntimeError("MLX_HOSTFILE environment variable must be set")
            return cls.singleton()

        hosts = _load_hostfile(hostfile)
        if len(hosts) == 1:
            return cls.singleton()

        rank_str = os.environ.get("MLX_RANK")
        if rank_str is None:
            if strict:
                raise RuntimeError("MLX_RANK environment variable must be set")
            rank = 0
        else:
            rank = int(rank_str)

        return cls.from_hosts(hosts, rank, connect_timeout=connect_timeout)

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size

    def split(self, color: int, key: int = -1) -> WinRingGroup:
        del color, key
        if self._size == 1:
            return WinRingGroup.singleton()
        raise NotImplementedError("WinRingGroup.split is not supported for size > 1")

    def _prune_completed_sends(self, *, raise_on_error: bool = True) -> None:
        still_running: list[Future[None]] = []
        first_err: Exception | None = None
        for future in self._send_futures:
            if future.done():
                exc = future.exception()
                if exc is not None and raise_on_error and first_err is None:
                    first_err = (
                        exc if isinstance(exc, Exception) else RuntimeError(str(exc))
                    )
            else:
                still_running.append(future)
        self._send_futures = still_running
        if first_err is not None:
            raise first_err

    def _drain_pending_sends(
        self,
        *,
        timeout: float | None = None,
        raise_on_error: bool = True,
    ) -> None:
        pending = self._send_futures
        self._send_futures = []
        deadline = None if timeout is None else time.monotonic() + timeout
        first_err: Exception | None = None
        for future in pending:
            wait_for = (
                None if deadline is None else max(0.0, deadline - time.monotonic())
            )
            try:
                future.result(timeout=wait_for)
            except TimeoutError:
                if raise_on_error and first_err is None:
                    raise
            except Exception as exc:
                if raise_on_error and first_err is None:
                    first_err = exc
        if first_err is not None:
            raise first_err

    def close(self) -> None:
        self._drain_pending_sends(timeout=5.0, raise_on_error=False)
        if self._sock_left is not None:
            try:
                self._sock_left.close()
            finally:
                self._sock_left = None
        if self._sock_right is not None:
            try:
                self._sock_right.close()
            finally:
                self._sock_right = None
        if self._send_executor is not None:
            self._send_executor.shutdown(wait=False)
            self._send_executor = None

    def _exchange_bytes(
        self,
        send_sock: socket.socket,
        send_mv: memoryview,
        recv_sock: socket.socket,
        recv_mv: memoryview,
    ) -> None:
        self._drain_pending_sends()
        if send_mv.nbytes == 0 and recv_mv.nbytes == 0:
            return
        if send_mv.nbytes == 0:
            _recv_exact(recv_sock, recv_mv)
            return
        if recv_mv.nbytes == 0:
            _send_exact(send_sock, send_mv)
            return

        executor = self._send_executor
        assert executor is not None
        send_future = executor.submit(_send_exact, send_sock, send_mv)
        _recv_exact(recv_sock, recv_mv)
        send_future.result()

    def send(self, x: mx.array, dst: int) -> mx.array:
        if self._size == 1 or dst == self._rank:
            raise RuntimeError(
                f"[ring] Cannot send from rank {self._rank} to self (size={self._size})"
            )
        right = (self._rank + 1) % self._size
        left = (self._rank + self._size - 1) % self._size
        if dst == right:
            sock = self._sock_right
        elif dst == left:
            sock = self._sock_left
        else:
            raise RuntimeError(
                f"[ring] Cannot send from rank {self._rank} to non-neighbor rank {dst} (size={self._size})"
            )
        assert sock is not None

        mx.eval(x)
        if x.nbytes > 0:
            payload = _array_to_bytes(x)
            executor = self._send_executor
            assert executor is not None
            self._prune_completed_sends(raise_on_error=True)
            self._send_futures.append(
                executor.submit(_send_exact, sock, memoryview(payload))
            )
        return x

    def recv(self, shape: Sequence[int], dtype: mx.Dtype, src: int) -> mx.array:
        if self._size == 1 or src == self._rank:
            raise RuntimeError(
                f"[ring] Cannot recv at rank {self._rank} from self (size={self._size})"
            )
        right = (self._rank + 1) % self._size
        left = (self._rank + self._size - 1) % self._size
        if src == left:
            sock = self._sock_left
        elif src == right:
            sock = self._sock_right
        else:
            raise RuntimeError(
                f"[ring] Cannot recv at rank {self._rank} from non-neighbor rank {src} (size={self._size})"
            )
        assert sock is not None

        shape_tuple = tuple(shape)
        numel = math.prod(shape_tuple) if shape_tuple else 1
        nbytes = numel * dtype.size
        if nbytes == 0:
            return mx.zeros(shape_tuple, dtype=dtype)

        buf = bytearray(nbytes)
        _recv_exact(sock, memoryview(buf))
        return _bytes_to_array(buf, dtype, shape_tuple)

    def recv_like(self, x: mx.array, src: int) -> mx.array:
        return self.recv(x.shape, x.dtype, src)

    def all_gather(self, x: mx.array) -> mx.array:
        output_shape = (
            (self._size,) if x.ndim == 0 else (x.shape[0] * self._size, *x.shape[1:])
        )
        if self._size == 1:
            mx.eval(x)
            return x.reshape(1) if x.ndim == 0 else x
        if x.nbytes == 0:
            return mx.zeros(output_shape, dtype=x.dtype)

        per_rank_bytes = x.nbytes
        buf = bytearray(per_rank_bytes * self._size)
        mv = memoryview(buf)

        local_start = self._rank * per_rank_bytes
        mv[local_start : local_start + per_rank_bytes] = _array_to_bytes(x)

        assert self._sock_left is not None and self._sock_right is not None
        send_segment = self._rank
        recv_segment = (self._rank + 1) % self._size

        # Matches ring.cpp:525, 756-790 (direction = +1).
        for _ in range(self._size - 1):
            send_off = send_segment * per_rank_bytes
            recv_off = recv_segment * per_rank_bytes
            self._exchange_bytes(
                self._sock_left,
                mv[send_off : send_off + per_rank_bytes],
                self._sock_right,
                mv[recv_off : recv_off + per_rank_bytes],
            )
            send_segment = (send_segment + 1) % self._size
            recv_segment = (recv_segment + 1) % self._size

        return _bytes_to_array(buf, x.dtype, output_shape)

    def _all_reduce(
        self,
        x: mx.array,
        reduce_op: Callable[[mx.array, mx.array], mx.array],
    ) -> mx.array:
        if self._size == 1 or x.size == 0:
            mx.eval(x)
            return x

        orig_shape = x.shape
        orig_size = x.size
        dtype = x.dtype
        flat = x.reshape(-1)
        if orig_size < self._size:
            pad = mx.zeros((self._size - orig_size,), dtype=dtype)
            flat = mx.concatenate([flat, pad], axis=0)
            direction = -1
        else:
            direction = 1
        mx.eval(flat)

        data_size = flat.size
        segment_size = (data_size + self._size - 1) // self._size
        bounds: list[tuple[int, int]] = []
        segments: list[mx.array] = []
        for seg_idx in range(self._size):
            start = min(seg_idx * segment_size, data_size)
            stop = min((seg_idx + 1) * segment_size, data_size)
            bounds.append((start, stop))
            segment = flat[start:stop]
            mx.eval(segment)
            segments.append(segment)

        assert self._sock_left is not None and self._sock_right is not None
        if direction == 1:
            send_sock = self._sock_left
            recv_sock = self._sock_right
            send_segment = self._rank
            recv_segment = (self._rank + 1) % self._size
        else:
            send_sock = self._sock_right
            recv_sock = self._sock_left
            send_segment = self._rank
            recv_segment = (self._rank - 1 + self._size) % self._size

        # Phase 1: ring reduce-scatter (size - 1 steps), matching ring.cpp:597-735.
        for _ in range(self._size - 1):
            send_bytes = _array_to_bytes(segments[send_segment])
            recv_count = bounds[recv_segment][1] - bounds[recv_segment][0]
            recv_buf = bytearray(recv_count * dtype.size)
            self._exchange_bytes(
                send_sock,
                memoryview(send_bytes),
                recv_sock,
                memoryview(recv_buf),
            )
            if recv_count > 0:
                rx_segment = _bytes_to_array(recv_buf, dtype, (recv_count,))
                reduced = reduce_op(segments[recv_segment], rx_segment)
                mx.eval(reduced)
                segments[recv_segment] = reduced
            send_segment = (send_segment + direction + self._size) % self._size
            recv_segment = (recv_segment + direction + self._size) % self._size

        # Phase 2: ring all-gather (size - 1 steps).
        for _ in range(self._size - 1):
            send_bytes = _array_to_bytes(segments[send_segment])
            recv_count = bounds[recv_segment][1] - bounds[recv_segment][0]
            recv_buf = bytearray(recv_count * dtype.size)
            self._exchange_bytes(
                send_sock,
                memoryview(send_bytes),
                recv_sock,
                memoryview(recv_buf),
            )
            if recv_count > 0:
                segments[recv_segment] = _bytes_to_array(recv_buf, dtype, (recv_count,))
            send_segment = (send_segment + direction + self._size) % self._size
            recv_segment = (recv_segment + direction + self._size) % self._size

        combined = mx.concatenate(segments, axis=0)
        if orig_size < self._size:
            combined = combined[:orig_size]
        result = mx.reshape(combined, orig_shape)
        mx.eval(result)
        return result

    def all_sum(self, x: mx.array) -> mx.array:
        return self._all_reduce(x, mx.add)

    def all_max(self, x: mx.array) -> mx.array:
        return self._all_reduce(x, mx.maximum)

    def all_min(self, x: mx.array) -> mx.array:
        return self._all_reduce(x, mx.minimum)


@dataclass(frozen=True)
class _OriginalDistributedOps:
    init: Callable[..., mx.distributed.Group]
    send: Callable[..., mx.array]
    recv: Callable[..., mx.array]
    recv_like: Callable[..., mx.array]
    all_gather: Callable[..., mx.array]
    all_sum: Callable[..., mx.array]
    all_max: Callable[..., mx.array]
    all_min: Callable[..., mx.array]


_originals: _OriginalDistributedOps | None = None
_default_group: WinRingGroup | None = None
_default_env_key: tuple[str | None, str | None] | None = None


def _as_mlx_group(group: WinRingGroup) -> mx.distributed.Group:
    return cast("mx.distributed.Group", cast("object", group))


def _get_or_init_group(
    group: mx.distributed.Group | None,
) -> WinRingGroup | mx.distributed.Group:
    group_obj = cast("object", group)
    if isinstance(group_obj, WinRingGroup):
        return group_obj
    if group is not None:
        return group
    if _default_group is not None:
        return _default_group
    return mx.distributed.init(strict=False, backend="any")


def install_win32_ring_patch() -> None:
    """Install the pure-Python Win32 TCP ring wrapper into ``mlx.core.distributed``."""
    global _originals
    if _originals is not None:
        return

    _originals = _OriginalDistributedOps(
        init=mx.distributed.init,
        send=mx.distributed.send,
        recv=mx.distributed.recv,
        recv_like=mx.distributed.recv_like,
        all_gather=mx.distributed.all_gather,
        all_sum=mx.distributed.all_sum,
        all_max=mx.distributed.all_max,
        all_min=mx.distributed.all_min,
    )
    originals = _originals

    def _patched_init(
        strict: bool = False,
        backend: str = "any",
    ) -> mx.distributed.Group:
        global _default_group, _default_env_key
        if backend not in ("ring", "any"):
            return originals.init(strict=strict, backend=backend)

        env_key = (os.environ.get("MLX_HOSTFILE"), os.environ.get("MLX_RANK"))
        if _default_group is not None:
            hostfile = env_key[0]
            if env_key == _default_env_key or not (
                hostfile and Path(hostfile).is_file()
            ):
                return _as_mlx_group(_default_group)

        group = WinRingGroup.from_env(strict=strict and backend == "ring")
        _default_group = group
        _default_env_key = env_key
        logger.debug(
            f"Initialized WinRingGroup (rank={group.rank()}, size={group.size()})"
        )
        return _as_mlx_group(group)

    def _patched_send(
        x: mx.array,
        dst: int,
        *,
        group: mx.distributed.Group | None = None,
        stream: mx.Stream | mx.Device | None = None,
    ) -> mx.array:
        resolved = _get_or_init_group(group)
        if isinstance(resolved, WinRingGroup):
            return resolved.send(x, dst)
        return originals.send(x, dst, group=resolved, stream=stream)

    def _patched_recv(
        shape: Sequence[int],
        dtype: mx.Dtype,
        src: int,
        *,
        group: mx.distributed.Group | None = None,
        stream: mx.Stream | mx.Device | None = None,
    ) -> mx.array:
        resolved = _get_or_init_group(group)
        if isinstance(resolved, WinRingGroup):
            return resolved.recv(shape, dtype, src)
        return originals.recv(shape, dtype, src, group=resolved, stream=stream)

    def _patched_recv_like(
        x: mx.array,
        src: int,
        *,
        group: mx.distributed.Group | None = None,
        stream: mx.Stream | mx.Device | None = None,
    ) -> mx.array:
        resolved = _get_or_init_group(group)
        if isinstance(resolved, WinRingGroup):
            return resolved.recv_like(x, src)
        return originals.recv_like(x, src, group=resolved, stream=stream)

    def _patched_all_gather(
        x: mx.array,
        *,
        group: mx.distributed.Group | None = None,
        stream: mx.Stream | mx.Device | None = None,
    ) -> mx.array:
        resolved = _get_or_init_group(group)
        if isinstance(resolved, WinRingGroup):
            return resolved.all_gather(x)
        return originals.all_gather(x, group=resolved, stream=stream)

    def _patched_all_sum(
        x: mx.array,
        *,
        group: mx.distributed.Group | None = None,
        stream: mx.Stream | mx.Device | None = None,
    ) -> mx.array:
        resolved = _get_or_init_group(group)
        if isinstance(resolved, WinRingGroup):
            return resolved.all_sum(x)
        return originals.all_sum(x, group=resolved, stream=stream)

    def _patched_all_max(
        x: mx.array,
        *,
        group: mx.distributed.Group | None = None,
        stream: mx.Stream | mx.Device | None = None,
    ) -> mx.array:
        resolved = _get_or_init_group(group)
        if isinstance(resolved, WinRingGroup):
            return resolved.all_max(x)
        return originals.all_max(x, group=resolved, stream=stream)

    def _patched_all_min(
        x: mx.array,
        *,
        group: mx.distributed.Group | None = None,
        stream: mx.Stream | mx.Device | None = None,
    ) -> mx.array:
        resolved = _get_or_init_group(group)
        if isinstance(resolved, WinRingGroup):
            return resolved.all_min(x)
        return originals.all_min(x, group=resolved, stream=stream)

    mx.distributed.init = _patched_init
    mx.distributed.send = _patched_send
    mx.distributed.recv = _patched_recv
    mx.distributed.recv_like = _patched_recv_like
    mx.distributed.all_gather = _patched_all_gather
    mx.distributed.all_sum = _patched_all_sum
    mx.distributed.all_max = _patched_all_max
    mx.distributed.all_min = _patched_all_min


def uninstall_win32_ring_patch() -> None:
    """Restore original ``mlx.core.distributed`` functions and reset cached group."""
    global _originals, _default_group, _default_env_key
    if _originals is not None:
        mx.distributed.init = _originals.init
        mx.distributed.send = _originals.send
        mx.distributed.recv = _originals.recv
        mx.distributed.recv_like = _originals.recv_like
        mx.distributed.all_gather = _originals.all_gather
        mx.distributed.all_sum = _originals.all_sum
        mx.distributed.all_max = _originals.all_max
        mx.distributed.all_min = _originals.all_min
        _originals = None
    if _default_group is not None:
        _default_group.close()
        _default_group = None
    _default_env_key = None
