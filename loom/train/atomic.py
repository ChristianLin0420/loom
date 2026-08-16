"""Atomic file operations.

Two invariants the whole checkpoint story rests on:

  1. never write a payload in place        -> write ``.tmp``, fsync, ``os.replace``
  2. never advance the LATEST pointer before its payload is durable

A SIGKILL at any instant must leave a loadable *earlier* checkpoint. R1/R2 run
3-6 days across dozens of 4 h links and will be preempted; a half-written
checkpoint that the next link happily loads is a silent month-long corruption.

Nothing here imports torch, so it is testable on a login node.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

__all__ = [
    "atomic_write_bytes", "atomic_write_text", "atomic_via_writer",
    "read_pointer", "write_pointer", "fsync_dir",
]


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


def atomic_write_text(path: str | Path, text: str) -> None:
    atomic_write_bytes(path, text.encode())


def atomic_via_writer(path: str | Path, writer: Callable[[str], None]) -> None:
    """For payloads written by a library (``torch.save``) rather than by us.

    The writer is handed a temporary path. Only after it returns does the file
    become visible under its real name.
    """
    path = Path(path)
    tmp = str(path) + ".tmp"
    writer(tmp)
    # torch.save does not fsync. Do it here, or `os.replace` publishes a name
    # whose data is still only in the page cache.
    fd = os.open(tmp, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)
    fsync_dir(path.parent)


def read_pointer(run_dir: str | Path, name: str = "LATEST") -> int | None:
    """Degrade to ``None`` on a missing or corrupt pointer. Never raise.

    A pointer that raises would take down every link of the chain; a pointer
    that returns None only costs the interval since the previous checkpoint.
    """
    p = Path(run_dir) / name
    if not p.exists():
        return None
    try:
        return int(p.read_text().strip())
    except (ValueError, OSError):
        return None


def write_pointer(run_dir: str | Path, step: int, name: str = "LATEST") -> None:
    atomic_write_text(Path(run_dir) / name, str(step))


def fsync_dir(d: str | Path) -> None:
    fd = os.open(str(d), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
