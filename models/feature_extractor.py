"""
================================================================================
DUAL-STREAM FEATURE EXTRACTOR
================================================================================

Module: Feature Extraction (Paper Section 3.4)

Two SEPARATE MobileNetV3-Large networks (NO weight sharing):
    - GlobalFeatureExtractor: Extracts global features from entire ROI
    - LocalFeatureExtractor: Extracts local features from diagonal patches

Input:
    - roi: (B, 3, 224, 224) - Global view
    - patches: (B, N, 3, 224, 224) - Local patches

Output:
    - global_features: (B, 960)
    - local_features: (B, N, 960)

================================================================================
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
from dataclasses import dataclass

# torchvision import for MobileNetV3
try:
    from torchvision.models import (
        mobilenet_v3_large,
        MobileNet_V3_Large_Weights
    )
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    print("Warning: torchvision not available. Install with: pip install torchvision")


@dataclass
class FeatureExtractorConfig:
    """Configuration for feature extractors."""
    pretrained: bool = True
    freeze_backbone: bool = False
    output_dim: int = 960  # MobileNetV3-Large final conv output


class GlobalFeatureExtractor(nn.Module):
    """
    Global Feature Extractor using MobileNetV3-Large.
    
    Extracts 960-dimensional feature vector from the entire upper-body ROI.
    
    Input: (B, 3, 224, 224)
    Output: (B, 960)
    """
    
    def __init__(self, config: Optional[FeatureExtractorConfig] = None):
        super().__init__()
        self.config = config or FeatureExtractorConfig()
        
        if not TORCHVISION_AVAILABLE:
            raise RuntimeError("torchvision is required for MobileNetV3")
        
        # Load MobileNetV3-Large
        if self.config.pretrained:
            weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
            backbone = mobilenet_v3_large(weights=weights)
        else:
            backbone = mobilenet_v3_large(weights=None)
        
        # Keep features and avgpool, remove classifier
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        
        # Optionally freeze backbone
        if self.config.freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
        
        self.output_dim = self.config.output_dim
    
    def forward(self, roi: torch.Tensor) -> torch.Tensor:
        """
        Extract global features from ROI.
        
        Args:
            roi: (B, 3, H, W) tensor
            
        Returns:
            features: (B, 960) tensor
        """
        x = self.features(roi)      # (B, 960, 7, 7) for 224x224 input
        x = self.avgpool(x)         # (B, 960, 1, 1)
        x = torch.flatten(x, 1)     # (B, 960)
        return x


class LocalFeatureExtractor(nn.Module):
    """
    Local Feature Extractor using MobileNetV3-Large.
    
    Extracts 960-dimensional feature vectors from each diagonal patch.
    IMPORTANT: Separate weights from GlobalFeatureExtractor (NO sharing).
    
    Input: (B, N, 3, 224, 224)
    Output: (B, N, 960)
    """
    
    def __init__(self, config: Optional[FeatureExtractorConfig] = None):
        super().__init__()
        self.config = config or FeatureExtractorConfig()
        
        if not TORCHVISION_AVAILABLE:
            raise RuntimeError("torchvision is required for MobileNetV3")
        
        # Load SEPARATE MobileNetV3-Large
        if self.config.pretrained:
            weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
            backbone = mobilenet_v3_large(weights=weights)
        else:
            backbone = mobilenet_v3_large(weights=None)
        
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        
        if self.config.freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
        
        self.output_dim = self.config.output_dim
    
    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Extract local features from patches.
        
        Args:
            patches: (B, N, 3, H, W) tensor
            
        Returns:
            features: (B, N, 960) tensor
        """
        B, N, C, H, W = patches.shape
        
        # Flatten batch and patches: (B*N, C, H, W)
        patches_flat = patches.view(B * N, C, H, W)
        
        # Extract features
        x = self.features(patches_flat)   # (B*N, 960, 7, 7)
        x = self.avgpool(x)               # (B*N, 960, 1, 1)
        x = torch.flatten(x, 1)           # (B*N, 960)
        
        # Reshape: (B, N, 960)
        features = x.view(B, N, -1)
        return features


class DualStreamFeatureExtractor(nn.Module):
    """
    Combined dual-stream feature extractor.
    
    Ensures two SEPARATE MobileNetV3-Large networks with independent weights.
    
    Input:
        - roi: (B, 3, 224, 224)
        - patches: (B, N, 3, 224, 224)
    
    Output:
        - global_features: (B, 960)
        - local_features: (B, N, 960)
    """
    
    def __init__(
        self,
        global_config: Optional[FeatureExtractorConfig] = None,
        local_config: Optional[FeatureExtractorConfig] = None
    ):
        super().__init__()
        
        self.global_extractor = GlobalFeatureExtractor(global_config)
        self.local_extractor = LocalFeatureExtractor(local_config)
        
        self.output_dim = self.global_extractor.output_dim
    
    def forward(
        self,
        roi: torch.Tensor,
        patches: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Extract both global and local features.
        
        Args:
            roi: (B, 3, H, W) tensor
            patches: (B, N, 3, H, W) tensor
            
        Returns:
            dict with 'global_features' and 'local_features'
        """
        global_features = self.global_extractor(roi)
        local_features = self.local_extractor(patches)
        
        return {
            'global_features': global_features,
            'local_features': local_features
        }


if __name__ == "__main__":
    # Quick test
    B, N = 2, 5
    roi = torch.rand(B, 3, 224, 224)
    patches = torch.rand(B, N, 3, 224, 224)
    
    extractor = DualStreamFeatureExtractor()
    output = extractor(roi, patches)
    
    print(f"Global features: {output['global_features'].shape}")
    print(f"Local features: {output['local_features'].shape}")
