from __future__ import annotations

from paguro.ashi.utils.terminal import TerminalDetector
from paguro.utils.dependencies import functools


@functools.cache
def terminal_detector() -> TerminalDetector:
    return TerminalDetector()
