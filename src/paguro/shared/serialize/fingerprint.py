from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


def combine_ordered(parts: Iterable[bytes], *, person=b"") -> bytes:
    """
    Combine digests where order matters (use for top-level assembly).

    Returns 32-byte (256-bit) hash - same security level as SHA256.
    """
    h = hashlib.blake2b(
        digest_size=32,  # 32 bytes = 256 bits of security
        person=b"fp:ord:" + person[:9],
    )
    parts = list(parts)
    h.update(len(parts).to_bytes(8, "big"))
    for d in parts:
        h.update(len(d).to_bytes(4, "big"))
        h.update(d)
    return h.digest()


def combine_unordered(parts: Iterable[bytes], *, person=b"") -> bytes:
    """
    Combine digests where order DOESN'T matter.

    Returns 32-byte (256-bit) hash - same security level as SHA256.
    """
    parts = sorted(parts)
    h = hashlib.blake2b(
        digest_size=32,  # 32 bytes = 256 bits of security
        person=b"fp:uno:" + person[:9],
    )
    h.update(len(parts).to_bytes(8, "big"))
    for d in parts:
        h.update(len(d).to_bytes(4, "big"))
        h.update(d)
    return h.digest()
