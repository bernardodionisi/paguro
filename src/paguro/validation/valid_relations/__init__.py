from __future__ import annotations

from paguro.validation.valid_relations._vrelations import VRelations

from paguro.validation.valid_relations.valid_pair import ValidPairRelation

vpair = ValidPairRelation
"""
Frame-Pair Validator
"""

__all__ = [
    "vpair",
    "VRelations",
    "ValidPairRelation",
]
