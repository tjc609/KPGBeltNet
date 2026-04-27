"""
================================================================================
BINARY CLASSIFIER
================================================================================

Module: Classification Head (Paper Section 3.7)

Binary classification head for seat belt detection.

Architecture:
    Dropout(0.5) → FC(512 → 128) → ReLU → Dropout(0.5) → FC(128 → 1)

Input:
    - sequence_encoding: (B, 512) - From Bi-GRU encoder

Output:
    - logits: (B, 1) - Raw logits for BCEWithLogitsLoss
    - probabilities: (B, 1) - Sigmoid probabilities

Labels:
    - 0: No seat belt (negative class)
    - 1: Wearing seat belt (positive class)

================================================================================
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class ClassifierConfig:
    """Configuration for binary classifier."""
    input_dim: int = 512              # From Bi-GRU (2 * hidden_dim)
    hidden_dim: int = 128             # Hidden layer dimension
    num_classes: int = 1              # 1 for BCE, 2 for CE
    dropout: float = 0.5              # Dropout rate
    use_hidden_layer: bool = True     # Include intermediate FC layer


class BinaryClassifier(nn.Module):
    """
    Binary Classification Head for seat belt detection.
    
    Architecture:
        Dropout → FC → ReLU → Dropout → FC → Logit
    
    Uses BCEWithLogitsLoss during training (numerically stable).
    Apply sigmoid to logits for probability output.
    """
    
    def __init__(self, config: Optional[ClassifierConfig] = None):
        super().__init__()
        self.config = config or ClassifierConfig()
        
        layers = []
        
        # Initial dropout
        layers.append(nn.Dropout(self.config.dropout))
        
        if self.config.use_hidden_layer:
            # Hidden layer
            layers.append(nn.Linear(self.config.input_dim, self.config.hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.config.dropout))
            
            # Output layer
            layers.append(nn.Linear(self.config.hidden_dim, self.config.num_classes))
        else:
            # Direct projection
            layers.append(nn.Linear(self.config.input_dim, self.config.num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, sequence_encoding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute classification logits.
        
        Args:
            sequence_encoding: (B, D_in)
            
        Returns:
            dict with:
                'logits': (B, 1)
                'probabilities': (B, 1)
        """
        logits = self.classifier(sequence_encoding)
        probabilities = torch.sigmoid(logits)
        
        return {
            'logits': logits,
            'probabilities': probabilities
        }


if __name__ == "__main__":
    # Quick test
    B = 4
    encoding = torch.rand(B, 512)
    
    classifier = BinaryClassifier()
    output = classifier(encoding)
    
    print(f"Input: {encoding.shape}")
    print(f"Logits: {output['logits'].shape}")
    print(f"Probabilities: {output['probabilities'].shape}")
    print(f"Sample probs: {output['probabilities'].squeeze().tolist()}")
