#!/usr/bin/env python3
"""
================================================================================
SEAT BELT DETECTION - INFERENCE & VISUALIZATION
================================================================================

Load trained model and visualize predictions on test images.

Usage:
    python scripts/inference.py --checkpoint runs/train_xxx/best_model.pth --image-dir data/test

================================================================================
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import SeatBeltDetector, SeatBeltDetectorConfig


# ==============================================================================
# DEFAULT CONFIGURATION
# ==============================================================================

CONFIG = {
    # Model checkpoint path
    'checkpoint': None,
    
    # Input: single image OR directory (set one, leave other as None)
    'image': None,  # e.g., '/path/to/single_image.jpg'
    'image_dir': None,
    
    # Output directory for saving results (set None to display only)
    'output_dir': str(PROJECT_ROOT / 'runs' / 'inference'),
    
    # Visualization options
    'show_attention': False,    # Show attention weights bar chart
    'show_patches': False,      # Show diagonal patches
    'grid': False,               # Show multiple images in grid view
    'max_images': 16,           # Maximum number of images (None = process ALL images)
    
    # Display options
    'no_display': False,        # If True, only save without displaying
}

# ==============================================================================


def parse_args() -> dict:
    """Parse command line arguments and return an inference config."""
    parser = argparse.ArgumentParser(description='Run classifier inference on ROI images')
    parser.add_argument('--checkpoint', required=True, help='Path to a trained checkpoint')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--image', help='Path to a single ROI image')
    source.add_argument('--image-dir', help='Directory containing ROI images')
    parser.add_argument('--output-dir', default=CONFIG['output_dir'],
                        help='Directory for rendered prediction images')
    parser.add_argument('--no-output', action='store_true',
                        help='Do not save results; display them instead')
    parser.add_argument('--show-attention', action='store_true',
                        help='Render attention weights')
    parser.add_argument('--show-patches', action='store_true',
                        help='Render sampled diagonal patches')
    parser.add_argument('--grid', action='store_true',
                        help='Render multiple images in one grid')
    parser.add_argument('--max-images', type=int, default=CONFIG['max_images'],
                        help='Maximum images to process from --image-dir; use 0 for all')
    args = parser.parse_args()

    cfg = CONFIG.copy()
    cfg.update({
        'checkpoint': args.checkpoint,
        'image': args.image,
        'image_dir': args.image_dir,
        'output_dir': None if args.no_output else args.output_dir,
        'show_attention': args.show_attention,
        'show_patches': args.show_patches,
        'grid': args.grid,
        'max_images': None if args.max_images == 0 else args.max_images,
    })
    return cfg

# Try to import matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")


# ==============================================================================
# IMAGE PREPROCESSING
# ==============================================================================

def get_transform():
    """Get inference transform (same as validation)."""
    import torchvision.transforms as T
    
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def load_image(image_path: str) -> Tuple[Image.Image, torch.Tensor]:
    """
    Load and preprocess image.
    
    Returns:
        original: PIL Image (for visualization)
        tensor: Preprocessed tensor (B, C, H, W)
    """
    image = Image.open(image_path).convert('RGB')
    transform = get_transform()
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image, tensor


def get_image_files(path: str, extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')) -> List[str]:
    """Get all image files from path (file or directory)."""
    path = Path(path)
    
    if path.is_file():
        return [str(path)]
    
    if path.is_dir():
        files = []
        for ext in extensions:
            files.extend(path.glob(f'*{ext}'))
            files.extend(path.glob(f'*{ext.upper()}'))
        return sorted([str(f) for f in files])
    
    return []


# ==============================================================================
# MODEL LOADING
# ==============================================================================

def load_model(checkpoint_path: str, device: torch.device) -> SeatBeltDetector:
    """Load trained model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config from checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
        model_config = SeatBeltDetectorConfig(
            n_patches=config.get('n_patches', 5),
            gru_hidden_dim=config.get('gru_hidden_dim', 256),
            classifier_dropout=config.get('classifier_dropout', 0.5)
        )
    else:
        model_config = SeatBeltDetectorConfig()
    
    # Create model
    model = SeatBeltDetector(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Print checkpoint info
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'best_f1' in checkpoint:
        print(f"  Best F1: {checkpoint['best_f1']:.4f}")
    if 'metrics' in checkpoint:
        metrics = checkpoint['metrics']
        print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
        print(f"  Precision: {metrics.get('precision', 0):.4f}")
        print(f"  Recall: {metrics.get('recall', 0):.4f}")
    
    return model


# ==============================================================================
# INFERENCE
# ==============================================================================

@torch.no_grad()
def predict_single(
    model: SeatBeltDetector,
    image_tensor: torch.Tensor,
    device: torch.device
) -> Dict:
    """
    Run inference on a single image.
    
    Returns:
        dict with prediction, probability, attention weights, etc.
    """
    image_tensor = image_tensor.to(device)
    
    output = model(image_tensor, return_attention=True, return_intermediate=True)
    
    probability = output['probabilities'].item()
    prediction = 1 if probability >= 0.5 else 0
    
    return {
        'prediction': prediction,
        'probability': probability,
        'label': 'WITH SEAT BELT' if prediction == 1 else 'NO SEAT BELT',
        'confidence': probability if prediction == 1 else 1 - probability,
        'attention_weights': output.get('attention_weights'),
        'patches': output.get('patches')
    }


def predict_batch(
    model: SeatBeltDetector,
    image_paths: List[str],
    device: torch.device
) -> List[Dict]:
    """Run inference on multiple images."""
    results = []
    
    for path in image_paths:
        try:
            original, tensor = load_image(path)
            result = predict_single(model, tensor, device)
            result['image_path'] = path
            result['original_image'] = original
            results.append(result)
        except Exception as e:
            print(f"Error processing {path}: {e}")
    
    return results


# ==============================================================================
# VISUALIZATION
# ==============================================================================

def visualize_single_result(
    result: Dict,
    save_path: Optional[str] = None,
    show_attention: bool = False,
    show_patches: bool = False
):
    """Visualize prediction result for a single image."""
    
    if not MATPLOTLIB_AVAILABLE:
        # Fallback: use PIL for basic visualization
        visualize_with_pil(result, save_path)
        return
    
    original = result['original_image']
    prediction = result['prediction']
    probability = result['probability']
    confidence = result['confidence']
    label = result['label']
    
    # Determine layout based on options
    n_cols = 1
    if show_attention and result.get('attention_weights') is not None:
        n_cols += 1
    if show_patches and result.get('patches') is not None:
        n_cols += 1
    
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]
    
    # Main image with prediction
    ax = axes[0]
    ax.imshow(original)
    
    # Color based on prediction
    color = 'green' if prediction == 1 else 'red'
    
    # Add prediction text
    title = f"{label}\nConfidence: {confidence:.1%}"
    ax.set_title(title, fontsize=14, fontweight='bold', color=color)
    ax.axis('off')
    
    # Add border
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(3)
    
    col_idx = 1
    
    # Attention weights visualization
    if show_attention and result.get('attention_weights') is not None:
        ax = axes[col_idx]
        attn = result['attention_weights'].squeeze().cpu().numpy()
        
        # Bar plot of attention weights
        n_patches = len(attn)
        bars = ax.bar(range(n_patches), attn, color='steelblue')
        
        # Highlight max attention
        max_idx = np.argmax(attn)
        bars[max_idx].set_color('coral')
        
        ax.set_xlabel('Patch Index')
        ax.set_ylabel('Attention Weight')
        ax.set_title('Part-to-Whole Attention')
        ax.set_xticks(range(n_patches))
        ax.set_xticklabels([f'P{i}\n({"shoulder" if i==0 else "hip" if i==n_patches-1 else "torso"})' 
                           for i in range(n_patches)], fontsize=8)
        col_idx += 1
    
    # Patches visualization
    if show_patches and result.get('patches') is not None:
        ax = axes[col_idx]
        patches_tensor = result['patches'].squeeze(0).cpu()  # (N, C, H, W)
        n_patches = patches_tensor.shape[0]
        
        # Create patch grid
        patch_grid = []
        for i in range(n_patches):
            patch = patches_tensor[i]
            # Denormalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            patch = patch * std + mean
            patch = patch.clamp(0, 1)
            patch_grid.append(patch.permute(1, 2, 0).numpy())
        
        # Show patches in a row
        combined = np.concatenate(patch_grid, axis=1)
        ax.imshow(combined)
        ax.set_title(f'Diagonal Patches (N={n_patches})')
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_with_pil(result: Dict, save_path: Optional[str] = None):
    """Fallback visualization using PIL."""
    original = result['original_image'].copy()
    prediction = result['prediction']
    confidence = result['confidence']
    label = result['label']
    
    # Resize for display
    display_size = (400, 400)
    original = original.resize(display_size, Image.LANCZOS)
    
    # Create draw object
    draw = ImageDraw.Draw(original)
    
    # Try to load font, fallback to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Color based on prediction
    color = (0, 200, 0) if prediction == 1 else (200, 0, 0)
    
    # Draw text background
    text = f"{label} ({confidence:.1%})"
    bbox = draw.textbbox((10, 10), text, font=font)
    draw.rectangle([bbox[0]-5, bbox[1]-5, bbox[2]+5, bbox[3]+5], fill=color)
    draw.text((10, 10), text, fill='white', font=font)
    
    # Draw border
    draw.rectangle([0, 0, display_size[0]-1, display_size[1]-1], outline=color, width=5)
    
    if save_path:
        original.save(save_path)
        print(f"Saved: {save_path}")
    else:
        original.show()


def visualize_grid(
    results: List[Dict],
    save_path: Optional[str] = None,
    max_images: int = 16,
    cols: int = 4
):
    """Visualize multiple results in a grid."""
    
    if not MATPLOTLIB_AVAILABLE:
        print("Grid visualization requires matplotlib")
        return
    
    n_images = min(len(results), max_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if n_images > 1 else [axes]
    
    for idx, (ax, result) in enumerate(zip(axes, results[:n_images])):
        original = result['original_image']
        prediction = result['prediction']
        confidence = result['confidence']
        label = result['label']
        
        ax.imshow(original)
        
        color = 'green' if prediction == 1 else 'red'
        ax.set_title(f"{label}\n{confidence:.1%}", fontsize=10, color=color)
        ax.axis('off')
        
        # Add border
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
            spine.set_visible(True)
    
    # Hide empty subplots
    for ax in axes[n_images:]:
        ax.axis('off')
    
    plt.suptitle(f'Seat Belt Detection Results ({n_images} images)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved grid: {save_path}")
    else:
        plt.show()
    
    plt.close()


def print_results_summary(results: List[Dict]):
    """Print summary of predictions."""
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS")
    print("=" * 60)
    
    with_belt = sum(1 for r in results if r['prediction'] == 1)
    without_belt = len(results) - with_belt
    
    print(f"\nTotal images: {len(results)}")
    print(f"  With seat belt: {with_belt}")
    print(f"  Without seat belt: {without_belt}")
    
    print("\nDetailed results:")
    print("-" * 60)
    
    for i, result in enumerate(results):
        path = Path(result['image_path']).name
        label = result['label']
        conf = result['confidence']
        status = "✓" if result['prediction'] == 1 else "✗"
        print(f"{i+1:3d}. {status} {path:40s} | {label:15s} | {conf:.1%}")


# ==============================================================================
# MAIN
# ==============================================================================

def main(cfg: dict):
    """Main inference function."""
    
    checkpoint_path = cfg['checkpoint']
    image_path = cfg.get('image')
    image_dir = cfg.get('image_dir')
    output_dir = cfg.get('output_dir')
    show_attention = cfg.get('show_attention', False)
    show_patches = cfg.get('show_patches', False)
    grid_view = cfg.get('grid', False)
    max_images = cfg.get('max_images', 16)
    
    # Validate inputs
    if image_path is None and image_dir is None:
        print("Error: pass --image or --image-dir")
        return
    
    # Get image paths
    if image_path:
        image_paths = get_image_files(image_path)
    else:
        image_paths = get_image_files(image_dir)
    
    if not image_paths:
        print("No images found!")
        return
    
    image_paths = image_paths[:max_images] if max_images else image_paths
    print(f"Found {len(image_paths)} images")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    model = load_model(checkpoint_path, device)
    
    # Run inference
    print("\nRunning inference...")
    results = predict_batch(model, image_paths, device)
    
    # Print summary
    print_results_summary(results)
    
    # Setup output directory
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None
    
    # Visualization
    if grid_view and len(results) > 1:
        save_path = output_path / 'grid_results.png' if output_path else None
        visualize_grid(results, save_path, max_images=max_images)
    else:
        for i, result in enumerate(results):
            if output_path:
                img_name = Path(result['image_path']).stem
                save_path = output_path / f'{img_name}_result.png'
            else:
                save_path = None
            
            visualize_single_result(
                result,
                save_path=str(save_path) if save_path else None,
                show_attention=show_attention,
                show_patches=show_patches
            )
    
    print("\nDone!")
    if output_path:
        print(f"Results saved to: {output_path}")


if __name__ == '__main__':
    main(parse_args())
