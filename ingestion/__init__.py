"""
Lazy Ingestion Pipeline
=======================

This module implements lazy ingestion:
- Lightweight CLIP embeddings at capture time
- Adaptive keyframe selection via semantic entropy
- FAISS index for fast L1 retrieval

Key classes:
- CLIPEncoder: Compute CLIP embeddings for images and text
- KeyframeSelector: Decide which frames to save
- FAISSIndexer: Build and query the semantic index
- MemoryLogger: Main orchestrator for ingestion
"""

from .clip_encoder import CLIPEncoder
from .keyframe_selector import KeyframeSelector, KeyframeDecision
from .faiss_indexer import FAISSIndexer
from .memory_logger import MemoryLogger, TraceLoader, KeyframeRecord

__all__ = [
    "CLIPEncoder",
    "KeyframeSelector",
    "KeyframeDecision",
    "FAISSIndexer",
    "MemoryLogger",
    "TraceLoader",
    "KeyframeRecord",
]
