"""
Evaluation Module
=================

Evaluate JIT Episodic Memory retrieval against ground truth.

Key scripts:
- full_eval_v2.py: Main evaluation on 181 HM3D scenes
- qualitative_figure.py: Generate success/failure visualization figures
- metrics.py: Core evaluation metric utilities
"""

from .metrics import (
    LocalizationResult,
    GroundTruthOracle,
    RetrievalEvaluator,
)

__all__ = [
    "LocalizationResult",
    "GroundTruthOracle",
    "RetrievalEvaluator",
]
