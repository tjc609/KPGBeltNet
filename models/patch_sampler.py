"""
================================================================================
DIAGONAL PATCH SAMPLER
================================================================================

Module: Diagonal Patch Sampling (Paper Section 3.3)

Samples M overlapping patches along the diagonal of the upper-body ROI,
from bottom-left (hip area) to top-right (shoulder area).

Parameters:
    - K: Hyper-parameter to control patch size (patch_size = H/K × W/K)
    - M: Number of patches to sample along the diagonal
    - Stride: (1 - sqrt(2/3)) × (W/K, H/K) to ensure ≥50% overlap

Input:
    - roi: (B, C, H, W) tensor - Upper-body ROI images

Output:
    - patches: (B, M, C, 299, 299) tensor
    - patch_coords: (B, M, 4) tensor [cx, cy, x1, y1]

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class PatchConfig:
    """Configuration for diagonal patch sampling."""
    K: int = 5                              # Patch scale factor (patch_size = H/K × W/K)
    M: int = 6                              # Number of patches along diagonal
    output_size: Tuple[int, int] = (224, 224)  # (H, W) resize for consistency
    diagonal_type: str = "main"             # "main" (anti-diagonal) or "both" (X pattern)


class DiagonalPatchSampler(nn.Module):
    """
    Diagonal Patch Sampler for seat belt detection.
    
    Samples patches along the anti-diagonal (bottom-left to top-right),
    which corresponds to the typical seat belt orientation across the torso.
    
    Spatial Interpretation:
        - Patch 0: Bottom-left (hip area)
        - Patch M-1: Top-right (shoulder area)
        - Middle patches: Chest/torso area (where seat belt is most visible)
    
    Overlap Calculation:
        - Stride coefficient: (1 - sqrt(2/3)) ≈ 0.1835
        - 1D overlap ratio: sqrt(2/3) ≈ 81.65%
        - 2D overlap area: 2/3 ≈ 66.7% (satisfies ≥50% requirement)
    """
    
    # Stride coefficient: (1 - sqrt(2/3)) for ≥50% overlap
    STRIDE_COEFF = 1 - math.sqrt(2 / 3)  # ≈ 0.1835
    
    def __init__(self, config: Optional[PatchConfig] = None):
        super().__init__()
        self.config = config or PatchConfig()
    
    @property
    def n_patches(self) -> int:
        return self.config.M
    
    @property
    def K(self) -> int:
        return self.config.K
    
    def _compute_patch_size(self, roi_size: Tuple[int, int]) -> Tuple[int, int]:
        """
        Compute patch size based on ROI dimensions and K.
        
        patch_size = (H/K, W/K)
        """
        H, W = roi_size
        ph = max(1, H // self.config.K)
        pw = max(1, W // self.config.K)
        return (ph, pw)
    
    def _compute_stride(self, patch_size: Tuple[int, int]) -> Tuple[float, float]:
        """
        Compute stride for diagonal sampling.
        
        stride = (1 - sqrt(2/3)) × (H/K, W/K)
        This ensures ≥50% overlap between adjacent patches.
        """
        ph, pw = patch_size
        stride_y = self.STRIDE_COEFF * ph
        stride_x = self.STRIDE_COEFF * pw
        return (stride_y, stride_x)
    
    def _compute_diagonal_centers(
        self,
        roi_size: Tuple[int, int],
        patch_size: Tuple[int, int],
        stride: Tuple[float, float],
        device: torch.device
    ) -> torch.Tensor:
        """
        Compute patch center coordinates along diagonal.
        
        Samples M patches from bottom-left to top-right using the stride.
        
        Args:
            roi_size: (H, W) of the ROI
            patch_size: (ph, pw) size of each patch
            stride: (stride_y, stride_x) step size
            device: Target device
            
        Returns:
            centers: (M, 2) tensor of (x, y) center coordinates
        """
        H, W = roi_size
        ph, pw = patch_size
        stride_y, stride_x = stride
        M = self.config.M
        
        centers = []
        
        # Start from bottom-left corner
        # Initial center position (with half patch offset from edge)
        start_x = pw / 2
        start_y = H - ph / 2
        
        for i in range(M):
            # Move along diagonal: right and up
            cx = start_x + i * stride_x
            cy = start_y - i * stride_y
            
            # Clamp to valid range
            cx = max(pw / 2, min(W - pw / 2, cx))
            cy = max(ph / 2, min(H - ph / 2, cy))
            
            centers.append([cx, cy])
        
        return torch.tensor(centers, device=device, dtype=torch.float32)
    
    def _extract_patches(
        self,
        roi: torch.Tensor,
        centers: torch.Tensor,
        patch_size: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract patches centered at given coordinates.
        
        Args:
            roi: (B, C, H, W) tensor
            centers: (M, 2) tensor of (x, y) center coordinates
            patch_size: (ph, pw) size of each patch
            
        Returns:
            patches: (B, M, C, H_out, W_out) tensor
            patch_coords: (B, M, 4) tensor of [cx, cy, x1, y1]
        """
        B, C, H, W = roi.shape
        M = centers.shape[0]
        ph, pw = patch_size
        oh, ow = self.config.output_size
        device = roi.device
        
        patches_list = []
        coords_list = []
        
        for i in range(M):
            cx, cy = centers[i]
            
            # Compute top-left corner
            x1 = int(cx - pw / 2)
            y1 = int(cy - ph / 2)
            
            # Clamp to valid range
            x1 = max(0, min(W - pw, x1))
            y1 = max(0, min(H - ph, y1))
            x2 = min(x1 + pw, W)
            y2 = min(y1 + ph, H)
            
            # Extract patch for all batches: (B, C, ph, pw)
            patch = roi[:, :, y1:y2, x1:x2]
            
            # Handle edge cases where patch might be smaller
            actual_h, actual_w = patch.shape[2], patch.shape[3]
            if actual_h != ph or actual_w != pw:
                patch = F.interpolate(patch, size=(ph, pw), mode='bilinear', align_corners=False)
            
            # Resize to output size (299 × 299)
            patch = F.interpolate(patch, size=(oh, ow), mode='bilinear', align_corners=False)
            
            patches_list.append(patch)
            coords_list.append(torch.tensor([cx.item(), cy.item(), x1, y1], device=device))
        
        # Stack: (B, M, C, oh, ow)
        patches = torch.stack(patches_list, dim=1)
        
        # Coords: (M, 4) -> expand to (B, M, 4)
        coords = torch.stack(coords_list, dim=0).unsqueeze(0).expand(B, -1, -1)
        
        return patches, coords
    
    def _normalize_patches(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Normalize patches using standard mean-std method.
        
        Args:
            patches: (B, M, C, H, W) tensor
            
        Returns:
            normalized patches: (B, M, C, H, W) tensor
        """
        # ImageNet mean and std
        mean = torch.tensor([0.485, 0.456, 0.406], device=patches.device)
        std = torch.tensor([0.229, 0.224, 0.225], device=patches.device)
        
        # Reshape for broadcasting: (1, 1, 3, 1, 1)
        mean = mean.view(1, 1, 3, 1, 1)
        std = std.view(1, 1, 3, 1, 1)
        
        # Normalize
        patches = (patches - mean) / std
        
        return patches
    
    def forward(
        self, 
        roi: torch.Tensor,
        normalize: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Sample diagonal patches from ROI.
        
        Args:
            roi: (B, C, H, W) or (C, H, W) tensor
            normalize: Whether to apply mean-std normalization
            
        Returns:
            dict with:
                - 'patches': (B, M, C, 299, 299) tensor
                - 'patch_coords': (B, M, 4) tensor [cx, cy, x1, y1]
                - 'n_patches': int (M)
                - 'patch_size': tuple (H/K, W/K)
        """
        # Handle unbatched input
        if roi.dim() == 3:
            roi = roi.unsqueeze(0)
        
        B, C, H, W = roi.shape
        device = roi.device
        
        # Compute patch size: (H/K, W/K)
        patch_size = self._compute_patch_size((H, W))
        
        # Compute stride: (1 - sqrt(2/3)) × patch_size
        stride = self._compute_stride(patch_size)
        
        # Compute diagonal centers
        centers = self._compute_diagonal_centers((H, W), patch_size, stride, device)
        
        # Extract patches
        patches, patch_coords = self._extract_patches(roi, centers, patch_size)
        
        # Apply normalization if requested
        if normalize:
            patches = self._normalize_patches(patches)
        
        return {
            'patches': patches,
            'patch_coords': patch_coords,
            'n_patches': self.config.M,
            'patch_size': patch_size,
            'K': self.config.K,
            'stride': stride
        }


if __name__ == "__main__":
    # Quick test
    print("=" * 60)
    print("Diagonal Patch Sampler Test")
    print("=" * 60)
    
    # Create sampler with paper's default: K=5, M=6
    config = PatchConfig(K=5, M=6, output_size=(299, 299))
    sampler = DiagonalPatchSampler(config)
    
    # Test with sample ROI
    roi = torch.rand(2, 3, 224, 224)
    output = sampler(roi, normalize=False)
    
    print(f"\nConfiguration:")
    print(f"  K (scale factor): {config.K}")
    print(f"  M (num patches): {config.M}")
    print(f"  Stride coefficient: {sampler.STRIDE_COEFF:.4f}")
    
    print(f"\nInput:")
    print(f"  ROI shape: {roi.shape}")
    
    print(f"\nOutput:")
    print(f"  Patches shape: {output['patches'].shape}")
    print(f"  Patch coords shape: {output['patch_coords'].shape}")
    print(f"  Computed patch size: {output['patch_size']}")
    print(f"  Computed stride: ({output['stride'][0]:.2f}, {output['stride'][1]:.2f})")
    
    # Verify overlap
    ph, pw = output['patch_size']
    stride_y, stride_x = output['stride']
    overlap_1d = 1 - (stride_x / pw)
    overlap_2d = overlap_1d ** 2
    print(f"\nOverlap verification:")
    print(f"  1D overlap ratio: {overlap_1d:.2%}")
    print(f"  2D overlap area: {overlap_2d:.2%}")
    print(f"  Satisfies ≥50%: {overlap_2d >= 0.5}")
