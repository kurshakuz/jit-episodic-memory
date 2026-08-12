# Dense Semantic Map Baseline
# ============================
#
# This module implements the "Dense Mapping" baseline that serves as a
# VLMaps/ConceptFusion-style external comparison for the JIT Cascade.
#
# Key insight: Uses SAME backbones (CLIP, OWL-ViT) as our method but
# differs in WHEN computation occurs:
#   - Dense Map: Build upfront (expensive), query fast
#   - JIT Cascade: Build lightweight index, compute at query time
#
# This isolates the architectural choice from backbone choice.

from .dense_map import DenseMapBaseline, JITBaselineWrapper
from .dense_map import MapBuildStats, MapQueryResult

__all__ = [
    "DenseMapBaseline",
    "JITBaselineWrapper", 
    "MapBuildStats",
    "MapQueryResult",
]
