"""
JIT Retrieval Cascade
=====================

The 3-level hierarchical retrieval system:

Level 1 - Semantic Filter (CLIP + FAISS):
    Fast approximate filtering using text-image similarity.
    Reduces N keyframes to ~100 candidates in O(1) time.

Level 2 - Geometric Clustering (Depth + DBSCAN):  
    Projects candidates to 3D, clusters nearby observations.
    Merges multiple views of same object into ~10 clusters.

Level 3 - Visual Verification (OWL-ViT):
    Zero-shot object detection on best frame per cluster.
    Confirms object presence with bounding boxes.

Main interface:
    from retrieval import JITRetrievalCascade
    
    cascade = JITRetrievalCascade("/path/to/trace")
    result = cascade.query("couch")
    if result.success:
        print(f"Found couch at {result.best_location.centroid_3d}")
"""

from .level1_semantic import Level1SemanticFilter, L1Candidate
from .level2_geometric import Level2GeometricCluster, L2Cluster, DepthProjector
from .level3_verification import Level3VisualVerification, L3VerifiedLocation, Detection, OWLViTDetector
from .cascade import JITRetrievalCascade, QueryResult

__all__ = [
    # Main interface
    "JITRetrievalCascade",
    "QueryResult",
    # Level 1
    "Level1SemanticFilter",
    "L1Candidate",
    # Level 2
    "Level2GeometricCluster",
    "L2Cluster",
    "DepthProjector",
    # Level 3
    "Level3VisualVerification",
    "L3VerifiedLocation",
    "Detection",
    "OWLViTDetector",
]
