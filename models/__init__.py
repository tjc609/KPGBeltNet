"""
Seat Belt Detection Pipeline - Model Components
================================================

Paper: "Seat Belt Detection using Part-to-Whole Attention 
        on Diagonally Sampled Patches"

Modules:
    - patch_sampler: Diagonal patch sampling from ROI
    - feature_extractor: Dual-stream MobileNetV3-Large (Global + Local)
    - attention: Part-to-Whole Attention mechanism
    - sequence_encoder: Bi-GRU temporal modeling
    - classifier: Binary classification head
    - pipeline: End-to-end SeatBeltDetector
"""

from .patch_sampler import DiagonalPatchSampler, PatchConfig
from .feature_extractor import (
    GlobalFeatureExtractor,
    LocalFeatureExtractor, 
    DualStreamFeatureExtractor,
    FeatureExtractorConfig
)
from .attention import PartToWholeAttention, PartToWholeAttentionConfig
from .sequence_encoder import BiGRUEncoder, BiGRUConfig
from .classifier import BinaryClassifier, ClassifierConfig
from .pipeline import SeatBeltDetector, SeatBeltDetectorConfig

__all__ = [
    # Patch Sampling
    'DiagonalPatchSampler',
    'PatchConfig',
    # Feature Extraction
    'GlobalFeatureExtractor',
    'LocalFeatureExtractor',
    'DualStreamFeatureExtractor',
    'FeatureExtractorConfig',
    # Attention
    'PartToWholeAttention',
    'PartToWholeAttentionConfig',
    # Sequence Encoding
    'BiGRUEncoder',
    'BiGRUConfig',
    # Classification
    'BinaryClassifier',
    'ClassifierConfig',
    # Full Pipeline
    'SeatBeltDetector',
    'SeatBeltDetectorConfig',
]
