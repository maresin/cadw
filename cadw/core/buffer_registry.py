"""v0.9.236: BufferRegistry — path-scoped buffer names, collision detection.

Anticipates pipeline step buffer naming (nested composite paths).
Collision on same (path, name) -> RuntimeError (no silent overwrite).

Validated in /tmp/lazy_proto.py (experiment H).
"""
from __future__ import annotations


class BufferRegistry:
    """Path-scoped buffer registry.

    Full name = f"{path}.{name}". Collision -> RuntimeError.
    Kind: "state" | "transient" | "materialized" (for accounting).
    """

    def __init__(self) -> None:
        self._buffers: dict[str, tuple[str, int]] = {}

    def allocate(self, path: str, name: str, nbytes: int, *,
                 kind: str) -> str:
        full = f"{path}.{name}"
        if full in self._buffers:
            raise RuntimeError(
                f"Buffer collision: {full!r} "
                f"(existing kind={self._buffers[full][0]!r})")
        self._buffers[full] = (kind, int(nbytes))
        return full

    def peak_state_bytes(self) -> int:
        return sum(sz for k, sz in self._buffers.values() if k == "state")

    def peak_transient_bytes(self) -> int:
        return sum(sz for k, sz in self._buffers.values()
                   if k == "transient")

    def peak_materialized_bytes(self) -> int:
        return sum(sz for k, sz in self._buffers.values()
                   if k == "materialized")

    def reset(self) -> None:
        self._buffers.clear()

    def dump(self) -> None:
        for path, (kind, sz) in sorted(self._buffers.items()):
            print(f"  {path:<58} {kind:<14} {sz:>8} B")
