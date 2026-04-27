#!/usr/bin/env python3
"""
================================================================================
SEAT BELT DETECTION - TRAINING SCRIPT
================================================================================

Paper: "Seat Belt Detection using Part-to-Whole Attention 
        on Diagonally Sampled Patches"

This is the FINAL training script for binary seat belt classification.

Features:
    - Dual-stream MobileNetV3 feature extraction
    - Part-to-Whole Attention mechanism
    - Bi-GRU sequence modeling
    - Mixed precision training (AMP)
    - Gradient clipping
    - Learning rate scheduling
    - Best model checkpoint (by F1 score)
    - Comprehensive metrics (Accuracy, Precision, Recall, F1)

Usage:
    $ python scripts/train.py --train-dirs data/train --val-dirs data/val

================================================================================
"""

import argparse
import os
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import SeatBeltDetector, SeatBeltDetectorConfig
from data import create_dataloaders


# ==============================================================================
# DEFAULT CONFIGURATION
# ==============================================================================

CONFIG = {
    'train_dirs': [str(PROJECT_ROOT / 'data' / 'train')],
    'val_dirs': [str(PROJECT_ROOT / 'data' / 'val')],
    
    # ===== 训练参数 =====
    'epochs': 50,               # 训练轮数
    'batch_size': 64,           # 批次大小
    'learning_rate': 1e-4,      # 学习率
    'weight_decay': 1e-2,       # AdamW 权重衰减
    'num_workers': 8,           # DataLoader 工作进程数
    
    # ===== 模型参数 =====
    'image_size': 224,          # 输入图像尺寸
    'n_patches': 6,             # 对角线采样块数
    'gru_hidden_dim': 256,      # Bi-GRU 隐藏维度 (每个方向)
    'classifier_dropout': 0.5,  # 分类器 Dropout 率
    
    # ===== 训练选项 =====
    'grad_clip': 1.0,           # 梯度裁剪最大范数
    'use_amp': True,            # 是否使用自动混合精度
    'seed': 42,                 # 随机种子
    'output_dir': None,         # None creates runs/train_YYYYmmdd_HHMMSS
}


def parse_args() -> dict:
    """Parse command line arguments and return a training config."""
    parser = argparse.ArgumentParser(description='Train the seat belt classifier')
    parser.add_argument('--train-dirs', nargs='+', default=CONFIG['train_dirs'],
                        help='Training directory/directories. Each can be a split root or a class folder.')
    parser.add_argument('--val-dirs', nargs='+', default=CONFIG['val_dirs'],
                        help='Validation directory/directories. Each can be a split root or a class folder.')
    parser.add_argument('--epochs', type=int, default=CONFIG['epochs'])
    parser.add_argument('--batch-size', type=int, default=CONFIG['batch_size'])
    parser.add_argument('--lr', type=float, default=CONFIG['learning_rate'])
    parser.add_argument('--weight-decay', type=float, default=CONFIG['weight_decay'])
    parser.add_argument('--num-workers', type=int, default=CONFIG['num_workers'])
    parser.add_argument('--image-size', type=int, default=CONFIG['image_size'])
    parser.add_argument('--n-patches', type=int, default=CONFIG['n_patches'])
    parser.add_argument('--gru-hidden-dim', type=int, default=CONFIG['gru_hidden_dim'])
    parser.add_argument('--classifier-dropout', type=float, default=CONFIG['classifier_dropout'])
    parser.add_argument('--grad-clip', type=float, default=CONFIG['grad_clip'])
    parser.add_argument('--seed', type=int, default=CONFIG['seed'])
    parser.add_argument('--output-dir', type=str, default=CONFIG['output_dir'])
    parser.add_argument('--no-amp', action='store_true', help='Disable automatic mixed precision')
    args = parser.parse_args()

    cfg = CONFIG.copy()
    cfg.update({
        'train_dirs': args.train_dirs,
        'val_dirs': args.val_dirs,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'num_workers': args.num_workers,
        'image_size': args.image_size,
        'n_patches': args.n_patches,
        'gru_hidden_dim': args.gru_hidden_dim,
        'classifier_dropout': args.classifier_dropout,
        'grad_clip': args.grad_clip,
        'seed': args.seed,
        'output_dir': args.output_dir,
        'use_amp': not args.no_amp,
    })
    return cfg


# ==============================================================================
# METRICS COMPUTATION
# ==============================================================================

class MetricsTracker:
    """Track and compute classification metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.predictions = []
        self.targets = []
        self.total_loss = 0.0
        self.num_batches = 0
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor, loss: float):
        """Update with batch predictions and targets."""
        # Convert logits to binary predictions
        binary_preds = (torch.sigmoid(preds) >= 0.5).long().view(-1)
        targets = targets.long().view(-1)
        
        self.predictions.extend(binary_preds.cpu().tolist())
        self.targets.extend(targets.cpu().tolist())
        self.total_loss += loss
        self.num_batches += 1
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        preds = torch.tensor(self.predictions)
        targets = torch.tensor(self.targets)
        
        # True positives, false positives, false negatives, true negatives
        tp = ((preds == 1) & (targets == 1)).sum().float()
        fp = ((preds == 1) & (targets == 0)).sum().float()
        fn = ((preds == 0) & (targets == 1)).sum().float()
        tn = ((preds == 0) & (targets == 0)).sum().float()
        
        # Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        
        # Precision: TP / (TP + FP)
        precision = tp / (tp + fp + 1e-8)
        
        # Recall: TP / (TP + FN)
        recall = tp / (tp + fn + 1e-8)
        
        # F1 Score
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        # Average loss
        avg_loss = self.total_loss / max(self.num_batches, 1)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'f1': f1.item(),
            'tp': tp.item(),
            'fp': fp.item(),
            'fn': fn.item(),
            'tn': tn.item()
        }


# ==============================================================================
# TRAINING FUNCTIONS
# ==============================================================================

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    grad_clip: float = 1.0,
    use_amp: bool = True
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: SeatBeltDetector model
        dataloader: Training dataloader
        criterion: BCEWithLogitsLoss
        optimizer: AdamW optimizer
        scaler: GradScaler for AMP
        device: Training device
        grad_clip: Gradient clipping max norm
        use_amp: Whether to use automatic mixed precision
        
    Returns:
        Dictionary of metrics
    """
    model.train()
    metrics = MetricsTracker()
    
    for batch_idx, batch in enumerate(dataloader):
        images = batch['image'].to(device)
        labels = batch['label'].to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        
        # Forward pass with AMP
        with autocast(enabled=use_amp):
            output = model(images, return_attention=False)
            logits = output['logits']
            loss = criterion(logits, labels)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        
        # Update metrics
        metrics.update(logits.detach(), batch['label'].to(device), loss.item())
        
        # Progress logging (every 50 batches)
        if (batch_idx + 1) % 50 == 0:
            current_metrics = metrics.compute()
            print(f"  Batch [{batch_idx + 1}/{len(dataloader)}] "
                  f"Loss: {current_metrics['loss']:.4f} "
                  f"Acc: {current_metrics['accuracy']:.4f}")
    
    return metrics.compute()


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True
) -> Dict[str, float]:
    """
    Validate model.
    
    Args:
        model: SeatBeltDetector model
        dataloader: Validation dataloader
        criterion: BCEWithLogitsLoss
        device: Training device
        use_amp: Whether to use automatic mixed precision
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    metrics = MetricsTracker()
    
    for batch in dataloader:
        images = batch['image'].to(device)
        labels = batch['label'].to(device).float().unsqueeze(1)
        
        with autocast(enabled=use_amp):
            output = model(images, return_attention=False)
            logits = output['logits']
            loss = criterion(logits, labels)
        
        metrics.update(logits, batch['label'].to(device), loss.item())
    
    return metrics.compute()


# ==============================================================================
# MAIN TRAINING LOOP
# ==============================================================================

def train(cfg: dict):
    """Main training function.
    
    Args:
        cfg: Configuration dictionary with all training parameters
    """
    
    # =========================================================================
    # Setup
    # =========================================================================
    
    # Set seed for reproducibility
    torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg['seed'])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print("SEAT BELT DETECTION TRAINING")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    cfg['use_amp'] = cfg['use_amp'] and device.type == 'cuda'
    
    # Output directory
    if cfg.get('output_dir'):
        output_dir = Path(cfg['output_dir'])
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / 'runs' / f'train_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Save configuration
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    
    # =========================================================================
    # Data
    # =========================================================================
    
    print(f"\n{'='*60}")
    print("LOADING DATA")
    print(f"{'='*60}")
    
    train_loader, val_loader = create_dataloaders(
        train_dirs=cfg['train_dirs'],
        val_dirs=cfg['val_dirs'],
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        image_size=cfg['image_size'],
        augment_train=True
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # =========================================================================
    # Model
    # =========================================================================
    
    print(f"\n{'='*60}")
    print("BUILDING MODEL")
    print(f"{'='*60}")
    
    model_config = SeatBeltDetectorConfig(
        n_patches=cfg['n_patches'],
        patch_size=(64, 64),
        patch_output_size=(224, 224),
        pretrained=True,
        freeze_backbone=False,
        feature_dim=960,
        attention_dropout=0.1,
        use_residual=True,
        use_layer_norm=True,
        gru_hidden_dim=cfg['gru_hidden_dim'],
        gru_num_layers=1,
        gru_dropout=0.1,
        classifier_hidden_dim=128,
        classifier_dropout=cfg['classifier_dropout']
    )
    
    model = SeatBeltDetector(model_config)
    model = model.to(device)
    
    # Print parameter counts
    params = model.count_parameters()
    print(f"\nModel Parameters:")
    for name, count in params.items():
        print(f"  {name}: {count:,}")
    
    # =========================================================================
    # Loss, Optimizer, Scheduler
    # =========================================================================
    
    criterion = nn.BCEWithLogitsLoss()
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg['learning_rate'],
        weight_decay=cfg['weight_decay']
    )
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg['epochs'],
        eta_min=cfg['learning_rate'] * 0.01
    )
    
    # Gradient scaler for AMP
    scaler = GradScaler(enabled=cfg['use_amp'])
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {cfg['epochs']}")
    print(f"  Batch size: {cfg['batch_size']}")
    print(f"  Learning rate: {cfg['learning_rate']}")
    print(f"  Weight decay: {cfg['weight_decay']}")
    print(f"  Gradient clipping: {cfg['grad_clip']}")
    print(f"  Mixed precision: {cfg['use_amp']}")
    
    # =========================================================================
    # Training Loop
    # =========================================================================
    
    print(f"\n{'='*60}")
    print("TRAINING")
    print(f"{'='*60}")
    
    best_f1 = 0.0
    best_epoch = 0
    history = {'train': [], 'val': []}
    
    for epoch in range(1, cfg['epochs'] + 1):
        epoch_start = time.time()
        
        print(f"\nEpoch {epoch}/{cfg['epochs']}")
        print("-" * 40)
        
        # Train
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            grad_clip=cfg['grad_clip'],
            use_amp=cfg['use_amp']
        )
        
        # Validate
        val_metrics = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=cfg['use_amp']
        )
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Log metrics
        epoch_time = time.time() - epoch_start
        
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f} | "
              f"Acc: {train_metrics['accuracy']:.4f} | "
              f"P: {train_metrics['precision']:.4f} | "
              f"R: {train_metrics['recall']:.4f} | "
              f"F1: {train_metrics['f1']:.4f}")
        
        print(f"Val   - Loss: {val_metrics['loss']:.4f} | "
              f"Acc: {val_metrics['accuracy']:.4f} | "
              f"P: {val_metrics['precision']:.4f} | "
              f"R: {val_metrics['recall']:.4f} | "
              f"F1: {val_metrics['f1']:.4f}")
        
        print(f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")
        
        # Save history
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)
        
        # Save best model (by F1 score)
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_epoch = epoch
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_f1': best_f1,
                'config': cfg,
                'metrics': val_metrics
            }
            
            torch.save(checkpoint, output_dir / 'best_model.pth')
            print(f"★ New best model saved! F1: {best_f1:.4f}")
        
        # Save latest model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'metrics': val_metrics
        }, output_dir / 'latest_model.pth')
    
    # =========================================================================
    # Training Complete
    # =========================================================================
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best F1: {best_f1:.4f} (Epoch {best_epoch})")
    print(f"Best model saved to: {output_dir / 'best_model.pth'}")
    
    # Save training history
    history_path = output_dir / 'history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")
    
    return best_f1


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == '__main__':
    train(parse_args())
