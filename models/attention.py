"""
================================================================================
PART-TO-WHOLE ATTENTION
================================================================================

Module: Part-to-Whole Attention (Paper Section 3.5)

Implements "Whole guides Parts" attention mechanism:
    - Query: Global ROI feature (the "Whole")
    - Key/Value: Local patch features (the "Parts")

The global context guides which local patches are most relevant,
enabling the model to focus on seat belt regions.

Input:
    - global_features: (B, 960) - From GlobalFeatureExtractor
    - local_features: (B, N, 960) - From LocalFeatureExtractor

Output:
    - co_attention_features: (B, N, 960) - Attended patch features
    - attention_weights: (B, 1, N) - For interpretability

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class PartToWholeAttentionConfig:
    """Configuration for Part-to-Whole Attention."""
    input_dim: int = 960          # MobileNetV3-Large output dimension
    attention_dim: int = 960      # Dimension for Q, K, V projections
    output_dim: int = 960         # Output dimension
    dropout: float = 0.1          # Dropout on attention weights
    use_residual: bool = True     # Include residual connection
    use_layer_norm: bool = True   # Layer normalization after attention


class PartToWholeAttention(nn.Module):
    """
    Part-to-Whole Attention: Global (Whole) attends to Local patches (Parts).
    
    Mechanism:
        1. Project global features to Query
        2. Project local features to Key and Value
        3. Scaled dot-product attention: softmax(QK^T / sqrt(d)) @ V
        4. Broadcast attended features to all patches
        5. Add residual connection + LayerNorm
    
    Data Flow:
        Global (B, D) → Q_proj → Q (B, 1, D)
        Local  (B, N, D) → K_proj → K (B, N, D)
        Local  (B, N, D) → V_proj → V (B, N, D)
        
        Attention: softmax(Q @ K^T / sqrt(d)) @ V → (B, 1, D)
        Broadcast: (B, 1, D) → (B, N, D)
        Output: broadcast + residual(Local) → LayerNorm → (B, N, D)
    """
    
    def __init__(self, config: Optional[PartToWholeAttentionConfig] = None):
        super().__init__()
        self.config = config or PartToWholeAttentionConfig()
        
        D_in = self.config.input_dim
        D_attn = self.config.attention_dim
        D_out = self.config.output_dim
        
        # Linear projections for Q, K, V
        self.W_query = nn.Linear(D_in, D_attn)
        self.W_key = nn.Linear(D_in, D_attn)
        self.W_value = nn.Linear(D_in, D_attn)
        
        # Output projection
        self.W_out = nn.Linear(D_attn, D_out)
        
        # Scaling factor for attention
        self.scale = math.sqrt(D_attn)
        
        # Dropout
        self.attn_dropout = nn.Dropout(self.config.dropout)
        
        # Layer normalization
        if self.config.use_layer_norm:
            self.layer_norm = nn.LayerNorm(D_out)
        else:
            self.layer_norm = nn.Identity()
        
        # Residual projection (if dimensions differ)
        if D_in != D_out:
            self.residual_proj = nn.Linear(D_in, D_out)
        else:
            self.residual_proj = nn.Identity()
        
        # Store attention weights for interpretability
        self._attention_weights = None
    
    def forward(
        self,
        global_features: torch.Tensor,
        local_features: torch.Tensor,
        return_attention: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Part-to-Whole attention.
        
        Args:
            global_features: (B, D_in) - Global ROI features
            local_features: (B, N, D_in) - Local patch features
            return_attention: Whether to return attention weights
            
        Returns:
            dict with:
                'co_attention_features': (B, N, D_out)
                'attention_weights': (B, 1, N) - Optional
        """
        B, N, D_in = local_features.shape
        
        # === Step 1: Project to Q, K, V ===
        Q = self.W_query(global_features).unsqueeze(1)  # (B, 1, D_attn)
        K = self.W_key(local_features)                   # (B, N, D_attn)
        V = self.W_value(local_features)                 # (B, N, D_attn)
        
        # === Step 2: Scaled Dot-Product Attention ===
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (B, 1, N)
        attn_weights = F.softmax(attn_scores, dim=-1)               # (B, 1, N)
        attn_weights_dropped = self.attn_dropout(attn_weights)
        
        # Store for interpretability
        self._attention_weights = attn_weights.detach()
        
        # === Step 3: Weighted sum of values ===
        attended = torch.bmm(attn_weights_dropped, V)  # (B, 1, D_attn)
        
        # === Step 4: Broadcast to all patches ===
        attended_broadcast = attended.expand(-1, N, -1)  # (B, N, D_attn)
        
        # === Step 5: Output projection ===
        output = self.W_out(attended_broadcast)  # (B, N, D_out)
        
        # === Step 6: Residual connection ===
        if self.config.use_residual:
            residual = self.residual_proj(local_features)
            output = output + residual
        
        # === Step 7: Layer normalization ===
        output = self.layer_norm(output)
        
        # Prepare result
        result = {'co_attention_features': output}
        if return_attention:
            result['attention_weights'] = attn_weights
        
        return result
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get last computed attention weights for visualization."""
        return self._attention_weights


if __name__ == "__main__":
    # Quick test
    B, N, D = 2, 5, 960
    global_features = torch.rand(B, D)
    local_features = torch.rand(B, N, D)
    
    attention = PartToWholeAttention()
    output = attention(global_features, local_features)
    
    print(f"Co-attention features: {output['co_attention_features'].shape}")
    print(f"Attention weights: {output['attention_weights'].shape}")
    print(f"Attention sums to 1: {output['attention_weights'].sum(dim=-1)}")
