#!/usr/bin/env python3
"""
LSeg Encoder for VLMaps

Wrapper around LSeg model to extract dense per-pixel CLIP-aligned features.

This provides the proper feature extraction for a faithful VLMaps implementation,
as described in Huang et al., ICRA 2023.

Reference:
    Li et al., "Language-driven Semantic Segmentation", ICLR 2022
    https://arxiv.org/abs/2201.03546
"""

import sys
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

# Add lang-seg to path
LSEG_PATH = Path(__file__).parent.parent / "lang-seg"


class LSegEncoder:
    """
    Dense CLIP-aligned encoder using LSeg.
    
    LSeg produces per-pixel embeddings aligned with CLIP text embeddings,
    used for VLMaps-style open-vocabulary localization.
    
    Key differences from CLIP patch tokens:
    1. True per-pixel resolution (not 32x32 patches)
    2. Dense decoder trained for semantic segmentation
    3. Better object boundary alignment
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        crop_size: int = 480,
    ):
        """
        Initialize LSeg encoder.
        
        Args:
            checkpoint_path: Path to LSeg checkpoint (demo_e200.ckpt)
            device: Device for inference
            crop_size: Crop size for LSeg (default 480)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.crop_size = crop_size
        self.embed_dim = 512  # LSeg output dimension
        
        if checkpoint_path is None:
            checkpoint_path = LSEG_PATH / "checkpoints" / "demo_e200.ckpt"
        self.checkpoint_path = Path(checkpoint_path)
        
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"LSeg checkpoint not found at {self.checkpoint_path}\n"
                "Download from: https://drive.google.com/file/d/1FTuHY1xPUkM-5gaDtMfgCl3D0gR89WV7"
            )
        
        self._model = None
        self._clip_model = None
        self._transform = None
        
    def _load_model(self):
        """Lazy load LSeg model directly from checkpoint."""
        if self._model is not None:
            return
            
        print(f"Loading LSeg model from {self.checkpoint_path}...")
        
        # Add lang-seg modules to path
        sys.path.insert(0, str(LSEG_PATH))
        
        # Import only what we need
        import clip
        from modules.models.lseg_net import LSegNet
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        
        # Get state dict (handle different checkpoint formats)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
            
        # Remove 'net.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("net."):
                new_state_dict[k[4:]] = v
            else:
                new_state_dict[k] = v
        
        # Get labels from ADE20K (needed for model init, but we won't use them)
        labels_file = LSEG_PATH / "label_files" / "ade20k_objectInfo150.txt"
        labels = []
        if labels_file.exists():
            import pandas as pd
            df = pd.read_csv(labels_file)
            labels = df["Name"].tolist()
        else:
            # Dummy labels
            labels = [f"class_{i}" for i in range(150)]
        
        # Create model
        self._model = LSegNet(
            labels=labels,
            backbone="clip_vitl16_384",
            features=256,
            crop_size=self.crop_size,
            arch_option=0,
            block_depth=0,
            activation='lrelu',
        )
        
        # Load weights
        self._model.load_state_dict(new_state_dict, strict=False)
        self._model.eval()
        self._model.to(self.device)
        
        # Store CLIP text encoder
        self._clip_model = self._model.clip_pretrained
        
        # Transform (LSeg uses 0.5 mean/std normalization)
        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        
        print(f"  LSeg loaded on {self.device}")
        
    def encode_image_dense(
        self,
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Get dense per-pixel CLIP-aligned embeddings.
        
        Args:
            image: RGB image as numpy array (H, W, 3), values 0-255
            target_size: Optional (H, W) to resize output features
            
        Returns:
            Dense embeddings (H', W', 512) where H', W' matches input or target_size
        """
        self._load_model()
        
        # Add lang-seg to path for imports
        sys.path.insert(0, str(LSEG_PATH))
        from modules.models.lseg_blocks import forward_vit
        
        orig_h, orig_w = image.shape[:2]
        
        # Convert to PIL and apply transform
        pil_image = Image.fromarray(image)
        img_tensor = self._transform(pil_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Forward through encoder to get dense features
            # Following LSeg architecture:
            # 1. ViT encoder with hooks
            # 2. Refinement network
            # 3. Head to 512-dim features
            
            x = img_tensor
            layer_1, layer_2, layer_3, layer_4 = forward_vit(self._model.pretrained, x)
            
            layer_1_rn = self._model.scratch.layer1_rn(layer_1)
            layer_2_rn = self._model.scratch.layer2_rn(layer_2)
            layer_3_rn = self._model.scratch.layer3_rn(layer_3)
            layer_4_rn = self._model.scratch.layer4_rn(layer_4)
            
            path_4 = self._model.scratch.refinenet4(layer_4_rn)
            path_3 = self._model.scratch.refinenet3(path_4, layer_3_rn)
            path_2 = self._model.scratch.refinenet2(path_3, layer_2_rn)
            path_1 = self._model.scratch.refinenet1(path_2, layer_1_rn)
            
            # Get dense 512-dim features
            image_features = self._model.scratch.head1(path_1)  # (B, 512, H', W')
            
            # Normalize
            image_features = F.normalize(image_features, dim=1)
            
            # Resize to target size or original size
            if target_size is not None:
                out_h, out_w = target_size
            else:
                out_h, out_w = orig_h, orig_w
                
            image_features = F.interpolate(
                image_features,
                size=(out_h, out_w),
                mode='bilinear',
                align_corners=True,
            )
            
            # Convert to (H, W, C)
            features = image_features[0].permute(1, 2, 0).cpu().numpy()
            
            return features
            
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text query to CLIP embedding.
        
        Args:
            text: Text query (e.g., "chair", "red couch")
            
        Returns:
            Text embedding (512,)
        """
        self._load_model()
        
        import clip
        
        with torch.no_grad():
            tokens = clip.tokenize([text]).to(self.device)
            text_features = self._clip_model.encode_text(tokens)
            text_features = F.normalize(text_features, dim=-1)
            return text_features[0].float().cpu().numpy()


def test_lseg_encoder():
    """Test LSeg encoder with a sample image."""
    print("Testing LSeg Encoder")
    
    # Create encoder
    encoder = LSegEncoder()
    
    # Create dummy image
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    print("\nEncoding image...")
    features = encoder.encode_image_dense(dummy_image)
    print(f"  Input shape: {dummy_image.shape}")
    print(f"  Output shape: {features.shape}")
    print(f"  Feature range: [{features.min():.3f}, {features.max():.3f}]")
    
    print("\nEncoding text...")
    text_feat = encoder.encode_text("chair")
    print(f"  Text feature shape: {text_feat.shape}")
    print(f"  Text feature norm: {np.linalg.norm(text_feat):.3f}")
    
    # Test similarity
    print("\nComputing similarity...")
    similarities = np.dot(features, text_feat)
    print(f"  Similarity map shape: {similarities.shape}")
    print(f"  Similarity range: [{similarities.min():.3f}, {similarities.max():.3f}]")
    
    print("\nLSeg encoder test passed")


if __name__ == "__main__":
    test_lseg_encoder()
