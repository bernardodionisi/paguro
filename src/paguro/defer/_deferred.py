from __future__ import annotations

from paguro.defer import LazyFrameExpr


class Deferred:

    def __call__(self, name: str | None = None) -> LazyFrameExpr:
        return LazyFrameExpr(name=name)

    def lazyframe(self, name: str | None = None) -> LazyFrameExpr:
        return LazyFrameExpr(name=name)

    def __getattr__(self, name: str) -> LazyFrameExpr:
        if name.startswith("__"):
            raise AttributeError(name)
        return LazyFrameExpr(name=name)
