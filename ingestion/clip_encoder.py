#!/usr/bin/env python3
"""
Phase 2: CLIP Encoder
=====================

Lightweight CLIP encoder for frame embeddings.
Uses OpenCLIP ViT-B-32 for fast inference.
"""

import numpy as np
import torch
from PIL import Image
from typing import Union, List, Optional


class CLIPEncoder:
    """OpenCLIP-based encoder for image and text embeddings."""
    
    def __init__(
        self,
        model_name: str = "ViT-B-32-quickgelu",
        pretrained: str = "laion400m_e32",
        device: Optional[str] = None,
    ):
        """
        Initialize CLIP encoder.
        
        Args:
            model_name: OpenCLIP model name
            pretrained: Pretrained weights to use
            device: Device to run on (auto-detect if None)
        """
        self.model_name = model_name
        self.pretrained = pretrained
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self._embedding_dim = None
        
    def load(self):
        """Load the CLIP model (lazy loading)."""
        if self.model is not None:
            return
            
        import open_clip
        
        print(f"Loading OpenCLIP {self.model_name} ({self.pretrained})...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, 
            pretrained=self.pretrained,
            device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.model.eval()
        
        # Get embedding dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=self.device)
            self._embedding_dim = self.model.encode_image(dummy).shape[-1]
            
        print(f"  Loaded on {self.device}, embedding dim: {self._embedding_dim}")
        
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        if self._embedding_dim is None:
            self.load()
        return self._embedding_dim
        
    def encode_image(
        self, 
        image: Union[np.ndarray, Image.Image, torch.Tensor],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode a single image to CLIP embedding.
        
        Args:
            image: RGB image (numpy HxWx3, PIL Image, or preprocessed tensor)
            normalize: Whether to L2-normalize the embedding
            
        Returns:
            Embedding vector of shape (embedding_dim,)
        """
        self.load()
        
        # Convert to PIL if needed
        if isinstance(image, np.ndarray):
            # Handle RGBA
            if image.shape[-1] == 4:
                image = image[:, :, :3]
            image = Image.fromarray(image.astype(np.uint8))
        
        # Preprocess
        if not isinstance(image, torch.Tensor):
            image = self.preprocess(image).unsqueeze(0).to(self.device)
            
        # Encode
        with torch.no_grad():
            embedding = self.model.encode_image(image)
            if normalize:
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                
        return embedding.cpu().numpy().squeeze()
    
    def encode_images(
        self,
        images: List[Union[np.ndarray, Image.Image]],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode multiple images to CLIP embeddings.
        
        Args:
            images: List of RGB images
            batch_size: Batch size for inference
            normalize: Whether to L2-normalize embeddings
            
        Returns:
            Embeddings of shape (N, embedding_dim)
        """
        self.load()
        
        all_embeddings = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            # Preprocess batch
            processed = []
            for img in batch:
                if isinstance(img, np.ndarray):
                    if img.shape[-1] == 4:
                        img = img[:, :, :3]
                    img = Image.fromarray(img.astype(np.uint8))
                processed.append(self.preprocess(img))
                
            batch_tensor = torch.stack(processed).to(self.device)
            
            # Encode
            with torch.no_grad():
                embeddings = self.model.encode_image(batch_tensor)
                if normalize:
                    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                    
            all_embeddings.append(embeddings.cpu().numpy())
            
        return np.vstack(all_embeddings)
    
    def encode_text(
        self,
        text: Union[str, List[str]],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode text query to CLIP embedding.
        
        Args:
            text: Single text string or list of strings
            normalize: Whether to L2-normalize the embedding
            
        Returns:
            Embedding(s) of shape (embedding_dim,) or (N, embedding_dim)
        """
        self.load()
        
        if isinstance(text, str):
            text = [text]
            squeeze = True
        else:
            squeeze = False
            
        # Tokenize
        tokens = self.tokenizer(text).to(self.device)
        
        # Encode
        with torch.no_grad():
            embeddings = self.model.encode_text(tokens)
            if normalize:
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                
        result = embeddings.cpu().numpy()
        
        if squeeze:
            return result.squeeze()
        return result
    
    def compute_similarity(
        self,
        image_embeddings: np.ndarray,
        text_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cosine similarity between image and text embeddings.
        
        Args:
            image_embeddings: Shape (N, D) or (D,)
            text_embeddings: Shape (M, D) or (D,)
            
        Returns:
            Similarity matrix of shape (N, M) or scalar
        """
        if image_embeddings.ndim == 1:
            image_embeddings = image_embeddings.reshape(1, -1)
        if text_embeddings.ndim == 1:
            text_embeddings = text_embeddings.reshape(1, -1)
            
        # Cosine similarity (embeddings should already be normalized)
        similarity = image_embeddings @ text_embeddings.T
        
        return similarity.squeeze()


# Simple test
if __name__ == "__main__":
    print("Testing CLIP Encoder...")
    
    encoder = CLIPEncoder()
    encoder.load()
    
    # Test with random image
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    embedding = encoder.encode_image(dummy_image)
    print(f"Image embedding shape: {embedding.shape}")
    print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
    
    # Test text encoding
    text_embedding = encoder.encode_text("a couch in a living room")
    print(f"Text embedding shape: {text_embedding.shape}")
    
    # Test similarity
    similarity = encoder.compute_similarity(embedding, text_embedding)
    print(f"Image-text similarity: {similarity:.4f}")
    
    print("[OK] CLIP encoder test passed!")
