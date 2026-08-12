#!/usr/bin/env python3
"""
Phase 2: FAISS Index Builder
============================

Builds and manages FAISS index for fast similarity search.
This is the Level 1 "Semantic Filter" component.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import json


class FAISSIndexer:
    """
    FAISS-based vector index for efficient similarity search.
    
    Supports both exact search (IndexFlatIP) and approximate search (IndexIVFFlat).
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        use_gpu: bool = False,
        index_type: str = "flat",  # "flat" or "ivf"
        nlist: int = 100,  # Number of clusters for IVF
    ):
        """
        Initialize FAISS indexer.
        
        Args:
            embedding_dim: Dimension of embeddings
            use_gpu: Whether to use GPU acceleration
            index_type: "flat" for exact search, "ivf" for approximate
            nlist: Number of clusters for IVF index
        """
        self.embedding_dim = embedding_dim
        self.use_gpu = use_gpu
        self.index_type = index_type
        self.nlist = nlist
        
        self.index = None
        self.frame_ids: List[int] = []  # Map index position to frame ID
        self.metadata: List[dict] = []  # Optional metadata per entry
        
        self._faiss = None
        
    def _get_faiss(self):
        """Lazy import of faiss."""
        if self._faiss is None:
            import faiss
            self._faiss = faiss
        return self._faiss
        
    def build_index(self, embeddings: np.ndarray, frame_ids: Optional[List[int]] = None):
        """
        Build the FAISS index from embeddings.
        
        Args:
            embeddings: Array of shape (N, embedding_dim)
            frame_ids: Optional list of frame IDs (defaults to 0, 1, 2, ...)
        """
        faiss = self._get_faiss()
        
        n_embeddings = embeddings.shape[0]
        
        if frame_ids is None:
            frame_ids = list(range(n_embeddings))
        self.frame_ids = frame_ids
        
        # Normalize embeddings for cosine similarity
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        
        if self.index_type == "flat":
            # Exact search with inner product (cosine sim after normalization)
            self.index = faiss.IndexFlatIP(self.embedding_dim)
        else:
            # Approximate search with IVF
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(
                quantizer, 
                self.embedding_dim, 
                min(self.nlist, n_embeddings),
                faiss.METRIC_INNER_PRODUCT,
            )
            # Need to train IVF index
            self.index.train(embeddings)
            
        self.index.add(embeddings)
        print(f"Built FAISS index with {self.index.ntotal} vectors")
        
    def add(self, embedding: np.ndarray, frame_id: int, metadata: Optional[dict] = None):
        """
        Add a single embedding to the index.
        
        Args:
            embedding: Single embedding vector
            frame_id: Frame ID for this embedding
            metadata: Optional metadata dict
        """
        faiss = self._get_faiss()
        
        if self.index is None:
            # Create index on first add
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            
        # Normalize for cosine similarity
        embedding = embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(embedding)
        
        self.index.add(embedding)
        self.frame_ids.append(frame_id)
        if metadata:
            self.metadata.append(metadata)
            
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 100,
    ) -> Tuple[List[int], List[float]]:
        """
        Search for most similar embeddings.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            
        Returns:
            Tuple of (frame_ids, similarities)
        """
        faiss = self._get_faiss()
        
        if self.index is None or self.index.ntotal == 0:
            return [], []
            
        # Normalize query
        query = query_embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query)
        
        # Search
        k = min(k, self.index.ntotal)
        similarities, indices = self.index.search(query, k)
        
        # Map indices to frame IDs
        result_frame_ids = [self.frame_ids[i] for i in indices[0]]
        result_similarities = similarities[0].tolist()
        
        return result_frame_ids, result_similarities
    
    
    def save(self, path: str):
        """Save index to disk."""
        faiss = self._get_faiss()
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path.with_suffix(".index")))
        
        # Save frame IDs and metadata
        meta = {
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type,
            "frame_ids": self.frame_ids,
            "metadata": self.metadata,
        }
        with open(path.with_suffix(".meta.json"), "w") as f:
            json.dump(meta, f)
            
        print(f"Saved FAISS index to {path}")
        
    def load(self, path: str):
        """Load index from disk."""
        faiss = self._get_faiss()
        
        path = Path(path)
        
        # Load FAISS index
        self.index = faiss.read_index(str(path.with_suffix(".index")))
        
        # Load metadata
        with open(path.with_suffix(".meta.json"), "r") as f:
            meta = json.load(f)
            
        self.embedding_dim = meta["embedding_dim"]
        self.index_type = meta["index_type"]
        self.frame_ids = meta["frame_ids"]
        self.metadata = meta.get("metadata", [])
        
        print(f"Loaded FAISS index with {self.index.ntotal} vectors")
        
    def __len__(self):
        if self.index is None:
            return 0
        return self.index.ntotal


# Simple test
if __name__ == "__main__":
    print("Testing FAISS Indexer...")
    
    indexer = FAISSIndexer(embedding_dim=512)
    
    # Create random embeddings
    np.random.seed(42)
    n_vectors = 100
    embeddings = np.random.randn(n_vectors, 512).astype(np.float32)
    
    # Build index
    indexer.build_index(embeddings)
    
    # Search with a query
    query = np.random.randn(512).astype(np.float32)
    frame_ids, similarities = indexer.search(query, k=5)
    
    print(f"Top 5 results:")
    for fid, sim in zip(frame_ids, similarities):
        print(f"  Frame {fid}: similarity={sim:.4f}")
        
    # Test save/load
    indexer.save("/tmp/test_faiss")
    
    indexer2 = FAISSIndexer(embedding_dim=512)
    indexer2.load("/tmp/test_faiss")
    
    frame_ids2, similarities2 = indexer2.search(query, k=5)
    assert frame_ids == frame_ids2, "Load/save mismatch!"
    
    print("[OK] FAISS indexer test passed!")
