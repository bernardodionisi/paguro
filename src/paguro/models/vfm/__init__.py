from __future__ import annotations

from typing import TypeVar

from paguro.models.vfm.vfmodel import (
    VFrameModel,
)
from paguro.models.vfm.utils import VFrameModelConfig
from paguro.models.vfm.decorators.transformed import transformed
from paguro.models.vfm.decorators.constraint import constraint
from paguro.models.vfm._blueprint import collect_model_blueprint

VFM = TypeVar("VFM", bound=VFrameModel)

__all__ = [
    "VFM",

    "VFrameModel",
    "VFrameModelConfig",
    "constraint",
    "transformed",

    "collect_model_blueprint",
]
