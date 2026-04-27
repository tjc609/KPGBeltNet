"""
================================================================================
SEAT BELT DETECTOR - END-TO-END PIPELINE
================================================================================

Complete pipeline integrating all modules:

    1. Diagonal Patch Sampling
    2. Dual-Stream Feature Extraction (Global + Local MobileNetV3)
    3. Part-to-Whole Attention
    4. Bi-GRU Sequence Encoder
    5. Binary Classifier

Paper: "Seat Belt Detection using Part-to-Whole Attention 
        on Diagonally Sampled Patches"

Input:
    - image: (B, 3, 224, 224) - Upper-body ROI (already cropped)

Output:
    - logits: (B, 1) - For BCEWithLogitsLoss
    - probabilities: (B, 1) - Sigmoid probabilities
    - attention_weights: (B, 1, N) - For interpretability

================================================================================
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
from dataclasses import dataclass

from .patch_sampler import DiagonalPatchSampler, PatchConfig
from .feature_extractor import DualStreamFeatureExtractor, FeatureExtractorConfig
from .attention import PartToWholeAttention, PartToWholeAttentionConfig
from .sequence_encoder import BiGRUEncoder, BiGRUConfig
from .classifier import BinaryClassifier, ClassifierConfig


@dataclass
class SeatBeltDetectorConfig:
    """Configuration for full SeatBeltDetector pipeline."""
    # Patch sampling
    n_patches: int = 5
    patch_size: tuple = (64, 64)
    patch_output_size: tuple = (224, 224)
    
    # Feature extraction
    pretrained: bool = True
    freeze_backbone: bool = False
    feature_dim: int = 960
    
    # Attention
    attention_dropout: float = 0.1
    use_residual: bool = True
    use_layer_norm: bool = True
    
    # Sequence encoder
    gru_hidden_dim: int = 256
    gru_num_layers: int = 1
    gru_dropout: float = 0.1
    
    # Classifier
    classifier_hidden_dim: int = 128
    classifier_dropout: float = 0.5


class SeatBeltDetector(nn.Module):
    """
    End-to-End Seat Belt Detector.
    
    Full Pipeline:
        Image (B, 3, 224, 224)
            │
            ├──► Patch Sampler ──► patches (B, N, 3, 224, 224)
            │                           │
            ▼                           ▼
        Global CNN                 Local CNN
        (B, 960)                   (B, N, 960)
            │                           │
            └───────────┬───────────────┘
                        │
                        ▼
              Part-to-Whole Attention
                   (B, N, 960)
                        │
                        ▼
                    Bi-GRU
                   (B, 512)
                        │
                        ▼
                   Classifier
                    (B, 1)
    """
    
    def __init__(self, config: Optional[SeatBeltDetectorConfig] = None):
        super().__init__()
        self.config = config or SeatBeltDetectorConfig()
        
        # === Module 1: Diagonal Patch Sampler ===
        patch_config = PatchConfig(
            M=self.config.n_patches,
            output_size=self.config.patch_output_size
        )
        self.patch_sampler = DiagonalPatchSampler(patch_config)
        
        # === Module 2: Dual-Stream Feature Extractor ===
        feature_config = FeatureExtractorConfig(
            pretrained=self.config.pretrained,
            freeze_backbone=self.config.freeze_backbone,
            output_dim=self.config.feature_dim
        )
        self.feature_extractor = DualStreamFeatureExtractor(
            global_config=feature_config,
            local_config=feature_config
        )
        
        # === Module 3: Part-to-Whole Attention ===
        attention_config = PartToWholeAttentionConfig(
            input_dim=self.config.feature_dim,
            attention_dim=self.config.feature_dim,
            output_dim=self.config.feature_dim,
            dropout=self.config.attention_dropout,
            use_residual=self.config.use_residual,
            use_layer_norm=self.config.use_layer_norm
        )
        self.attention = PartToWholeAttention(attention_config)
        
        # === Module 4: Bi-GRU Encoder ===
        gru_config = BiGRUConfig(
            input_dim=self.config.feature_dim,
            hidden_dim=self.config.gru_hidden_dim,
            num_layers=self.config.gru_num_layers,
            dropout=self.config.gru_dropout,
            bidirectional=True,
            aggregation="last"
        )
        self.sequence_encoder = BiGRUEncoder(gru_config)
        
        # === Module 5: Binary Classifier ===
        classifier_config = ClassifierConfig(
            input_dim=self.sequence_encoder.output_dim,  # 512
            hidden_dim=self.config.classifier_hidden_dim,
            num_classes=1,
            dropout=self.config.classifier_dropout,
            use_hidden_layer=True
        )
        self.classifier = BinaryClassifier(classifier_config)
    
    def forward(
        self,
        image: torch.Tensor,
        return_attention: bool = True,
        return_intermediate: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through entire pipeline.
        
        Args:
            image: (B, 3, H, W) - Upper-body ROI image
            return_attention: Whether to return attention weights
            return_intermediate: Whether to return intermediate representations
            
        Returns:
            dict with:
                'logits': (B, 1)
                'probabilities': (B, 1)
                'attention_weights': (B, 1, N) - if return_attention
                'global_features': (B, 960) - if return_intermediate
                'local_features': (B, N, 960) - if return_intermediate
                'co_attention_features': (B, N, 960) - if return_intermediate
                'sequence_encoding': (B, 512) - if return_intermediate
        """
        # === Step 1: Diagonal Patch Sampling ===
        patch_output = self.patch_sampler(image)
        patches = patch_output['patches']  # (B, N, 3, H, W)
        
        # === Step 2: Dual-Stream Feature Extraction ===
        # Global: process full image, Local: process patches
        features = self.feature_extractor(image, patches)
        global_features = features['global_features']  # (B, 960)
        local_features = features['local_features']    # (B, N, 960)
        
        # === Step 3: Part-to-Whole Attention ===
        attention_output = self.attention(
            global_features, 
            local_features,
            return_attention=return_attention
        )
        co_attention_features = attention_output['co_attention_features']  # (B, N, 960)
        
        # === Step 4: Bi-GRU Sequence Encoding ===
        encoder_output = self.sequence_encoder(co_attention_features)
        sequence_encoding = encoder_output['sequence_encoding']  # (B, 512)
        
        # === Step 5: Binary Classification ===
        classifier_output = self.classifier(sequence_encoding)
        
        # === Prepare output ===
        result = {
            'logits': classifier_output['logits'],
            'probabilities': classifier_output['probabilities']
        }
        
        if return_attention:
            result['attention_weights'] = attention_output.get('attention_weights')
        
        if return_intermediate:
            result['global_features'] = global_features
            result['local_features'] = local_features
            result['co_attention_features'] = co_attention_features
            result['sequence_encoding'] = sequence_encoding
            result['patches'] = patches
        
        return result
    
    def count_parameters(self) -> Dict[str, int]:
        """Count trainable parameters for each component."""
        def count_params(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        return {
            'patch_sampler': count_params(self.patch_sampler),
            'global_extractor': count_params(self.feature_extractor.global_extractor),
            'local_extractor': count_params(self.feature_extractor.local_extractor),
            'attention': count_params(self.attention),
            'sequence_encoder': count_params(self.sequence_encoder),
            'classifier': count_params(self.classifier),
            'total': count_params(self)
        }


if __name__ == "__main__":
    print("=" * 60)
    print("Testing SeatBeltDetector Pipeline")
    print("=" * 60)
    
    # Create model
    model = SeatBeltDetector()
    
    # Create dummy input
    B = 2
    image = torch.rand(B, 3, 224, 224)
    
    print(f"\nInput shape: {image.shape}")
    
    # Forward pass
    output = model(image, return_attention=True, return_intermediate=True)
    
    print(f"\nOutput shapes:")
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    # Parameter count
    print(f"\nParameter counts:")
    params = model.count_parameters()
    for name, count in params.items():
        print(f"  {name}: {count:,}")
    
    print(f"\nTotal trainable parameters: {params['total']:,}")
