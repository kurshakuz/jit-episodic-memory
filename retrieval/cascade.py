#!/usr/bin/env python3
"""
JIT Retrieval Cascade
=====================

The complete 3-level retrieval cascade.

Levels:
1. Semantic Filter (CLIP + FAISS): 100 candidates
2. Geometric Clustering (Depth + DBSCAN): 10 clusters
3. Visual Verification (OWL-ViT): 3 verified locations

Total latency: ~250ms (within 500ms budget for robotics)
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion import TraceLoader, CLIPEncoder
from retrieval.level1_semantic import Level1SemanticFilter
from retrieval.level2_geometric import Level2GeometricCluster
from retrieval.level3_verification import Level3VisualVerification, L3VerifiedLocation


@dataclass
class QueryResult:
    """Complete result of a JIT retrieval query."""
    query: str
    success: bool
    locations: List[L3VerifiedLocation]
    best_location: Optional[L3VerifiedLocation]
    
    # Timing breakdown
    l1_time_ms: float
    l2_time_ms: float
    l3_time_ms: float
    total_time_ms: float
    
    # Pipeline stats
    l1_candidates: int
    l2_clusters: int
    l3_verified: int
    
    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "success": self.success,
            "best_location": self.best_location.to_dict() if self.best_location else None,
            "num_locations": len(self.locations),
            "timing": {
                "l1_ms": self.l1_time_ms,
                "l2_ms": self.l2_time_ms,
                "l3_ms": self.l3_time_ms,
                "total_ms": self.total_time_ms,
            },
            "pipeline_stats": {
                "l1_candidates": self.l1_candidates,
                "l2_clusters": self.l2_clusters,
                "l3_verified": self.l3_verified,
            },
        }


class JITRetrievalCascade:
    """
    Just-in-Time Episodic Memory Retrieval Cascade.
    
    This is the main interface for querying the robot's memory.
    It orchestrates the 3-level pipeline for accurate and fast retrieval.
    
    Example usage:
        cascade = JITRetrievalCascade("/path/to/trace")
        result = cascade.query("where is the couch?")
        if result.success:
            location = result.best_location.centroid_3d
            print(f"Couch is at {location}")
    """
    
    def __init__(
        self,
        trace_dir: str,
        # L1 config
        l1_k_candidates: int = 100,
        # L2 config
        l2_eps: float = 1.0,
        l2_min_samples: int = 1,
        l2_top_clusters: int = 10,
        # L3 config
        l3_max_verify: int = 5,
        l3_threshold: float = 0.1,
        # Auto-L3: automatically enable/disable L3 based on frame density
        auto_l3: bool = False,
        auto_l3_spacing_threshold: float = 0.5,  # meters
        # General
        lazy_load_models: bool = True,
        owl_detector=None,
        clip_encoder=None,
    ):
        """
        Initialize JIT Retrieval Cascade.

        Args:
            trace_dir: Path to exploration trace
            l1_k_candidates: Number of L1 candidates to retrieve
            l2_eps: DBSCAN clustering radius
            l2_min_samples: Min observations per cluster
            l2_top_clusters: Max clusters from L2
            l3_max_verify: Max clusters to verify in L3
            l3_threshold: Min detection score for verification
            auto_l3: If True, enable/disable L3 based on median frame spacing
            auto_l3_spacing_threshold: Enable L3 when median spacing < this (m)
            lazy_load_models: If True, load models on first query
            owl_detector: Pre-loaded OWLViTDetector to reuse across scenes
            clip_encoder: Pre-loaded CLIPEncoder to reuse across scenes
        """
        self.trace_dir = trace_dir
        self.auto_l3 = auto_l3
        self.auto_l3_spacing_threshold = auto_l3_spacing_threshold
        self.config = {
            "l1_k_candidates": l1_k_candidates,
            "l2_eps": l2_eps,
            "l2_min_samples": l2_min_samples,
            "l2_top_clusters": l2_top_clusters,
            "l3_max_verify": l3_max_verify,
            "l3_threshold": l3_threshold,
        }

        # Load trace
        print(f"Loading trace from: {trace_dir}")
        self.trace_loader = TraceLoader(trace_dir)

        # Compute median inter-frame spacing for auto-L3
        self._median_spacing = None
        self._l3_auto_enabled = None
        if self.auto_l3:
            self._median_spacing = self._compute_median_spacing()
            self._l3_auto_enabled = self._median_spacing < self.auto_l3_spacing_threshold
            print(f"  Auto-L3: median spacing = {self._median_spacing:.3f}m, "
                  f"L3 {'enabled' if self._l3_auto_enabled else 'disabled'}")
        
        # Create pipeline components
        self.clip_encoder = clip_encoder or (CLIPEncoder() if not lazy_load_models else None)
        
        self.l1_filter = None
        self.l2_cluster = None
        self.l3_verify = None
        self._owl_detector = owl_detector
        
        self._initialized = False
        
        if not lazy_load_models:
            self._initialize_pipeline()
            
    def _compute_median_spacing(self) -> float:
        """Compute median Euclidean distance between consecutive keyframes."""
        trace = self.trace_loader.trace
        if len(trace) < 2:
            return float('inf')
        positions = trace[['x', 'y', 'z']].values.astype(float)
        diffs = np.diff(positions, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        return float(np.median(distances))

    def _initialize_pipeline(self):
        """Initialize pipeline components."""
        if self._initialized:
            return
            
        # Create CLIP encoder if needed
        if self.clip_encoder is None:
            self.clip_encoder = CLIPEncoder()
            
        # L1: Semantic Filter
        self.l1_filter = Level1SemanticFilter(
            trace_loader=self.trace_loader,
            clip_encoder=self.clip_encoder,
            k_candidates=self.config["l1_k_candidates"],
        )
        
        # L2: Geometric Clustering
        self.l2_cluster = Level2GeometricCluster(
            trace_loader=self.trace_loader,
            eps=self.config["l2_eps"],
            min_samples=self.config["l2_min_samples"],
            top_clusters=self.config["l2_top_clusters"],
        )
        
        # L3: Visual Verification
        self.l3_verify = Level3VisualVerification(
            trace_loader=self.trace_loader,
            detector=self._owl_detector,
            verification_threshold=self.config["l3_threshold"],
        )
        
        self._initialized = True
        
    def query(
        self,
        query: str,
        skip_l3: bool = False,
        synonyms: Optional[List[str]] = None,
    ) -> QueryResult:
        """
        Query the episodic memory for an object.

        Args:
            query: Natural language query (e.g., "couch", "red chair")
            skip_l3: If True, skip L3 verification (faster but less accurate).
                     When auto_l3 is enabled, this flag is overridden by the
                     automatic decision based on median frame spacing.
            synonyms: Alternative query terms for better recall

        Returns:
            QueryResult with locations and timing information
        """
        # Initialize if needed
        self._initialize_pipeline()

        # Auto-L3: override skip_l3 based on frame density
        if self.auto_l3 and self._l3_auto_enabled is not None:
            skip_l3 = not self._l3_auto_enabled

        start_total = time.time()
        
        # L1: Semantic Filter
        start = time.time()
        l1_candidates = self.l1_filter.retrieve(query)
        l1_time = (time.time() - start) * 1000
        
        # L2: Geometric Clustering
        start = time.time()
        l2_clusters = self.l2_cluster.cluster(l1_candidates)
        l2_time = (time.time() - start) * 1000
        
        # L3: Visual Verification
        l3_time = 0.0
        if skip_l3:
            # Convert L2 clusters to "unverified" L3 locations
            locations = [
                L3VerifiedLocation(
                    cluster_id=c.cluster_id,
                    frame_id=c.best_frame_id,
                    centroid_3d=c.centroid,
                    detections=[],
                    best_detection_score=c.max_similarity,
                    verified=False,  # Not verified
                )
                for c in l2_clusters
            ]
        else:
            start = time.time()
            if synonyms:
                locations = self.l3_verify.verify_with_synonyms(
                    l2_clusters, query, synonyms,
                    max_verify=self.config["l3_max_verify"],
                )
            else:
                locations = self.l3_verify.verify(
                    l2_clusters, query,
                    max_verify=self.config["l3_max_verify"],
                )
            l3_time = (time.time() - start) * 1000
            
        total_time = (time.time() - start_total) * 1000
        
        # Determine success and best location
        verified_locations = [loc for loc in locations if loc.verified]
        success = len(verified_locations) > 0
        best_location = verified_locations[0] if verified_locations else None
        
        # If no verified locations but we have clusters, use best cluster
        if best_location is None and locations:
            best_location = locations[0]
            success = False  # Not verified, but have candidate
            
        return QueryResult(
            query=query,
            success=success,
            locations=locations,
            best_location=best_location,
            l1_time_ms=l1_time,
            l2_time_ms=l2_time,
            l3_time_ms=l3_time,
            total_time_ms=total_time,
            l1_candidates=len(l1_candidates),
            l2_clusters=len(l2_clusters),
            l3_verified=len(verified_locations),
        )
    
    def query_fast(self, query: str) -> QueryResult:
        """
        Fast query that skips L3 verification.
        
        Use this when speed is critical and lower precision is acceptable.
        Latency: ~50ms (vs ~250ms with L3)
        """
        return self.query(query, skip_l3=True)
    
    
def interactive_demo():
    """Interactive demo of the JIT retrieval system."""
    
    # Find trace
    trace_dir = Path(__file__).parent.parent / "outputs" / "phase2" / "exploration_panoramic"
    if not trace_dir.exists():
        trace_dir = Path(__file__).parent.parent / "outputs" / "phase2" / "exploration"
        
    print("=" * 60)
    print("JIT Episodic Memory - Interactive Demo")
    print("=" * 60)
    print()
    
    # Create cascade
    cascade = JITRetrievalCascade(str(trace_dir))
    
    print(f"\nLoaded memory with {len(cascade.trace_loader.trace)} keyframes")
    print()
    print("Enter queries like 'couch', 'table', 'window'")
    print("Commands: 'fast:<query>' for L1+L2 only, 'quit' to exit")
    print()
    
    while True:
        try:
            user_input = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
            
        if not user_input:
            continue
            
        if user_input.lower() == "quit":
            break
            
        # Check for fast mode
        skip_l3 = False
        if user_input.lower().startswith("fast:"):
            skip_l3 = True
            user_input = user_input[5:].strip()
            
        # Query
        result = cascade.query(user_input, skip_l3=skip_l3)
        
        # Display results
        print()
        print(f"Query: '{result.query}'")
        print(f"  Success: {'[OK]' if result.success else '[FAIL]'}")
        print(f"  Pipeline: {result.l1_candidates} -> {result.l2_clusters} -> {result.l3_verified}")
        print(f"  Timing: L1={result.l1_time_ms:.1f}ms, L2={result.l2_time_ms:.1f}ms, "
              f"L3={result.l3_time_ms:.1f}ms")
        print(f"  Total: {result.total_time_ms:.1f}ms")
        
        if result.best_location:
            loc = result.best_location
            print(f"  Best location:")
            print(f"    Frame: {loc.frame_id}")
            print(f"    3D position: [{loc.centroid_3d[0]:.2f}, {loc.centroid_3d[1]:.2f}, "
                  f"{loc.centroid_3d[2]:.2f}]")
            if loc.verified:
                print(f"    Detection score: {loc.best_detection_score:.3f}")
                
        print()
    
    print("\nGoodbye!")


def benchmark():
    """Benchmark the complete pipeline."""
    
    trace_dir = Path(__file__).parent.parent / "outputs" / "phase2" / "exploration_panoramic"
    if not trace_dir.exists():
        trace_dir = Path(__file__).parent.parent / "outputs" / "phase2" / "exploration"
        
    print("=" * 60)
    print("JIT Retrieval Cascade Benchmark")
    print("=" * 60)
    print()
    
    cascade = JITRetrievalCascade(str(trace_dir))
    
    # Warm up
    print("Warming up...")
    cascade.query("couch")
    cascade.query("table")
    
    # Test queries
    queries = ["couch", "chair", "table", "window", "door", "lamp", "plant"]
    
    print("\n=== Full Pipeline (L1 + L2 + L3) ===\n")
    
    full_times = []
    for query in queries:
        times = []
        for _ in range(3):
            result = cascade.query(query)
            times.append(result.total_time_ms)
            
        avg_time = np.mean(times)
        full_times.append(avg_time)
        
        status = "[OK]" if result.success else "[FAIL]"
        print(f"{status} '{query}': {avg_time:.1f}ms "
              f"({result.l1_candidates}->{result.l2_clusters}->{result.l3_verified})")
    
    print(f"\nAverage full pipeline: {np.mean(full_times):.1f}ms")
    
    print("\n=== Fast Pipeline (L1 + L2 only) ===\n")
    
    fast_times = []
    for query in queries:
        times = []
        for _ in range(10):
            result = cascade.query_fast(query)
            times.append(result.total_time_ms)
            
        avg_time = np.mean(times)
        fast_times.append(avg_time)
        
        print(f"'{query}': {avg_time:.1f}ms "
              f"({result.l1_candidates}->{result.l2_clusters})")
    
    print(f"\nAverage fast pipeline: {np.mean(fast_times):.1f}ms")
    
    print("\n=== Summary ===")
    print(f"Full pipeline: {np.mean(full_times):.1f}ms (±{np.std(full_times):.1f})")
    print(f"Fast pipeline: {np.mean(fast_times):.1f}ms (±{np.std(fast_times):.1f})")
    print(f"Speedup: {np.mean(full_times)/np.mean(fast_times):.1f}x")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        benchmark()
    else:
        interactive_demo()
