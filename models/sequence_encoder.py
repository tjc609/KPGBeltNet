"""
================================================================================
BI-GRU SEQUENCE ENCODER
================================================================================

Module: Sequence Modeling (Paper Section 3.6)

Bidirectional GRU for modeling sequential relationships between diagonal patches.

The sequence represents spatial ordering along the diagonal:
    - Position 0: Top-left patch (shoulder area)
    - Position N-1: Bottom-right patch (hip area)

Bi-GRU captures context in both directions, important because:
    - Seat belt runs from shoulder to hip
    - Visibility may vary along the diagonal

Input:
    - co_attention_features: (B, N, 960) - From Part-to-Whole Attention

Output:
    - sequence_encoding: (B, 512) - Aggregated representation

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Literal
from dataclasses import dataclass


@dataclass
class BiGRUConfig:
    """Configuration for Bi-GRU encoder."""
    input_dim: int = 960              # From attention output
    hidden_dim: int = 256             # Hidden size per direction
    num_layers: int = 1               # Number of GRU layers
    dropout: float = 0.1              # Dropout between layers
    bidirectional: bool = True        # Use bidirectional GRU
    aggregation: Literal["last", "mean", "max", "attention"] = "last"


class BiGRUEncoder(nn.Module):
    """
    Bidirectional GRU Encoder for sequential patch features.
    
    Architecture:
        Bi-GRU: (B, N, 960) → (B, N, 512)
        Aggregation: (B, N, 512) → (B, 512)
    
    Aggregation methods:
        - "last": Concatenate final hidden states from both directions
        - "mean": Mean pooling over all timesteps
        - "max": Max pooling over all timesteps
        - "attention": Learned attention pooling
    """
    
    def __init__(self, config: Optional[BiGRUConfig] = None):
        super().__init__()
        self.config = config or BiGRUConfig()
        
        # Bi-GRU layer
        self.gru = nn.GRU(
            input_size=self.config.input_dim,
            hidden_size=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            batch_first=True,
            dropout=self.config.dropout if self.config.num_layers > 1 else 0,
            bidirectional=self.config.bidirectional
        )
        
        # Output dimension
        self.num_directions = 2 if self.config.bidirectional else 1
        self.output_dim = self.config.hidden_dim * self.num_directions  # 512
        
        # Attention pooling (optional)
        if self.config.aggregation == "attention":
            self.attn_weights = nn.Linear(self.output_dim, 1)
    
    def _aggregate_last(
        self, 
        output: torch.Tensor, 
        hidden: torch.Tensor
    ) -> torch.Tensor:
        """
        Aggregate using final hidden states from both directions.
        
        For bidirectional GRU:
            hidden[-2]: Forward final hidden (after seeing patch N-1)
            hidden[-1]: Backward final hidden (after seeing patch 0)
        """
        if self.config.bidirectional:
            forward_hidden = hidden[-2]   # (B, H)
            backward_hidden = hidden[-1]  # (B, H)
            return torch.cat([forward_hidden, backward_hidden], dim=-1)
        else:
            return hidden[-1]
    
    def _aggregate_mean(self, output: torch.Tensor) -> torch.Tensor:
        """Mean pooling over sequence dimension."""
        return output.mean(dim=1)
    
    def _aggregate_max(self, output: torch.Tensor) -> torch.Tensor:
        """Max pooling over sequence dimension."""
        return output.max(dim=1)[0]
    
    def _aggregate_attention(self, output: torch.Tensor) -> torch.Tensor:
        """Learned attention pooling."""
        scores = self.attn_weights(output)        # (B, N, 1)
        weights = F.softmax(scores, dim=1)        # (B, N, 1)
        return (weights * output).sum(dim=1)      # (B, 2*H)
    
    def forward(
        self,
        co_attention_features: torch.Tensor,
        return_all_hidden: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Encode sequence of patch features.
        
        Args:
            co_attention_features: (B, N, D_in)
            return_all_hidden: Whether to return all timestep hidden states
            
        Returns:
            dict with:
                'sequence_encoding': (B, 512)
                'all_hidden': (B, N, 512) - Optional
        """
        # GRU forward pass
        output, hidden = self.gru(co_attention_features)
        # output: (B, N, 512), hidden: (2*num_layers, B, H)
        
        # Aggregate sequence
        if self.config.aggregation == "last":
            sequence_encoding = self._aggregate_last(output, hidden)
        elif self.config.aggregation == "mean":
            sequence_encoding = self._aggregate_mean(output)
        elif self.config.aggregation == "max":
            sequence_encoding = self._aggregate_max(output)
        elif self.config.aggregation == "attention":
            sequence_encoding = self._aggregate_attention(output)
        else:
            raise ValueError(f"Unknown aggregation: {self.config.aggregation}")
        
        result = {'sequence_encoding': sequence_encoding}
        if return_all_hidden:
            result['all_hidden'] = output
        
        return result


if __name__ == "__main__":
    # Quick test
    B, N, D = 2, 5, 960
    features = torch.rand(B, N, D)
    
    encoder = BiGRUEncoder()
    output = encoder(features)
    
    print(f"Input: {features.shape}")
    print(f"Sequence encoding: {output['sequence_encoding'].shape}")
    print(f"Output dim: {encoder.output_dim}")
