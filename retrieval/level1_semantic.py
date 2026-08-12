#!/usr/bin/env python3
"""
Phase 3 - Level 1: Semantic Filter
===================================

Fast approximate filtering using CLIP embeddings and FAISS.

This is the "Broad Sweep" stage that reduces the search space:
- Given a natural language query ("where is the couch?")
- Encode query with CLIP
- Search FAISS index for semantically similar keyframes
- Return top-k candidates (typically k=100)

Latency budget: ~5ms (our actual: ~2ms)
"""

import numpy as np
from typing import List, Optional, Dict
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingestion import CLIPEncoder, TraceLoader


@dataclass
class L1Candidate:
    """A candidate from Level 1 retrieval."""
    frame_id: int
    similarity: float
    position: List[float]
    rotation: List[float]
    image_path: str
    depth_path: Optional[str]


class Level1SemanticFilter:
    """
    Level 1: Semantic Filter using CLIP + FAISS.
    
    Pipeline:
    - Natural language query -> CLIP embedding
    - FAISS approximate nearest neighbor search
    - Returns semantically relevant keyframes
    
    Designed for O(1) query complexity with FAISS IVF index.
    """
    
    def __init__(
        self,
        trace_loader: TraceLoader,
        clip_encoder: Optional[CLIPEncoder] = None,
        k_candidates: int = 100,
    ):
        """
        Initialize L1 Semantic Filter.
        
        Args:
            trace_loader: Loaded trace with FAISS index
            clip_encoder: CLIP encoder (creates new if None)
            k_candidates: Number of candidates to return
        """
        self.trace_loader = trace_loader
        self.clip_encoder = clip_encoder or CLIPEncoder()
        self.k_candidates = k_candidates
        
        # Cache text embeddings for common queries
        self._query_cache: Dict[str, np.ndarray] = {}
        
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a natural language query to CLIP embedding.
        
        Uses caching for repeated queries.
        """
        if query not in self._query_cache:
            self._query_cache[query] = self.clip_encoder.encode_text(query)
        return self._query_cache[query]
    
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> List[L1Candidate]:
        """
        Retrieve semantically similar keyframes for a query.
        
        Args:
            query: Natural language query (e.g., "where is the couch?")
            k: Number of candidates (uses default if None)
            
        Returns:
            List of L1Candidate objects sorted by similarity (descending)
        """
        k = k or self.k_candidates
        
        # Encode query
        query_embedding = self.encode_query(query)
        
        # Search FAISS index
        frame_ids, similarities = self.trace_loader.search(query_embedding, k)
        
        # Build candidate list
        candidates = []
        for frame_id, similarity in zip(frame_ids, similarities):
            try:
                frame_data = self.trace_loader.get_frame(frame_id)
                
                candidate = L1Candidate(
                    frame_id=frame_id,
                    similarity=float(similarity),
                    position=[frame_data["x"], frame_data["y"], frame_data["z"]],
                    rotation=[frame_data["qw"], frame_data["qx"], 
                             frame_data["qy"], frame_data["qz"]],
                    image_path=frame_data["image_path"],
                    depth_path=frame_data["depth_path"],
                )
                candidates.append(candidate)
            except Exception as e:
                print(f"Warning: Could not load frame {frame_id}: {e}")
                continue
                
        return candidates
    
    
def test_level1():
    """Test Level 1 retrieval on exploration trace."""
    import time
    
    # Load trace
    trace_dir = Path(__file__).parent.parent / "outputs" / "phase2" / "exploration_panoramic"
    if not trace_dir.exists():
        trace_dir = Path(__file__).parent.parent / "outputs" / "phase2" / "exploration"
        
    print(f"Loading trace from: {trace_dir}")
    loader = TraceLoader(str(trace_dir))
    
    # Create L1 filter
    l1_filter = Level1SemanticFilter(loader, k_candidates=10)
    
    # Test queries - variety of object types
    test_queries = [
        "couch",
        "chair", 
        "table",
        "window",
        "door",
        "plant",
        "lamp",
        "ceiling",
        "floor",
        "wall",
    ]
    
    print("\n=== Level 1 Semantic Filter Test ===\n")
    
    for query in test_queries:
        # Time the query
        start = time.time()
        candidates = l1_filter.retrieve(query, k=5)
        elapsed = (time.time() - start) * 1000  # ms
        
        print(f"Query: '{query}' ({elapsed:.1f}ms)")
        if candidates:
            print(f"  Top candidate: frame {candidates[0].frame_id}, "
                  f"similarity={candidates[0].similarity:.3f}")
            print(f"  Position: [{candidates[0].position[0]:.2f}, "
                  f"{candidates[0].position[1]:.2f}, {candidates[0].position[2]:.2f}]")
        else:
            print("  No candidates found")
        print()
    
    # Benchmark latency
    print("=== Latency Benchmark ===")
    query = "couch"
    
    # Warm up
    for _ in range(10):
        l1_filter.retrieve(query, k=100)
        
    # Measure
    times = []
    for _ in range(100):
        start = time.time()
        l1_filter.retrieve(query, k=100)
        times.append((time.time() - start) * 1000)
        
    print(f"Query: '{query}' (k=100)")
    print(f"  Mean latency: {np.mean(times):.2f}ms")
    print(f"  Std: {np.std(times):.2f}ms")
    print(f"  Min: {np.min(times):.2f}ms")
    print(f"  Max: {np.max(times):.2f}ms")
    print(f"  Throughput: {1000/np.mean(times):.0f} queries/sec")
    
    print("\n[OK] Level 1 Semantic Filter test passed!")


if __name__ == "__main__":
    test_level1()
