"""
JIT Episodic Memory for Mobile Robots
======================================

A 3-level hierarchical retrieval system for efficient object localization
in robot episodic memory.

Modules:
    - ingestion: Memory building (CLIP encoding, FAISS indexing)
    - retrieval: 3-level retrieval cascade (L1 semantic, L2 geometric, L3 visual)
    - evaluation: Evaluation scripts and metrics
    - oracle: Ground truth generation

Example:
    >>> from retrieval import JITRetrievalCascade
    >>> cascade = JITRetrievalCascade("outputs/<scene>/exploration")
    >>> result = cascade.query("Where is the couch?")
    >>> if result.success:
    ...     print(f"Found at: {result.best_location.centroid_3d}")
"""

__version__ = "1.0.0"
__author__ = "Shyngyskhan Abilkassov, Almas Shintemirov"
__license__ = "MIT"
