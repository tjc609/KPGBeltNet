#!/usr/bin/env python3
"""
================================================================================
SEAT BELT DETECTION - CONFUSION MATRIX EVALUATION
================================================================================

计算并可视化安全带分类算法的混淆矩阵。

混淆矩阵 (Confusion Matrix) 包含:
    - TP (True Positive): 预测为佩戴安全带，实际也是佩戴安全带
    - TN (True Negative): 预测为未佩戴安全带，实际也是未佩戴安全带
    - FP (False Positive): 预测为佩戴安全带，实际未佩戴安全带
    - FN (False Negative): 预测为未佩戴安全带，实际佩戴安全带

Usage:
    $ python scripts/evaluate_confusion_matrix.py --model-path runs/train_xxx/best_model.pth --eval-dirs data/val

================================================================================
"""

import argparse
import os
import sys
import json
from pathlib import Path
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models import SeatBeltDetector, SeatBeltDetectorConfig
from data import create_dataloaders

# 尝试导入可视化库
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False
    print("警告: 未安装 matplotlib 或 seaborn，将跳过可视化。")
    print("安装方法: pip install matplotlib seaborn")


# ==============================================================================
# DEFAULT CONFIGURATION
# ==============================================================================

CONFIG = {
    'model_path': None,  # 如果为 None，将自动查找最新的 best_model.pth
    'eval_dirs': [str(PROJECT_ROOT / 'data' / 'val')],
    'batch_size': 64,
    'num_workers': 8,
    'image_size': 224,
    'save_plot': True,          # 是否保存混淆矩阵图片
    'show_plot': False,         # 是否显示图片 (服务器上建议关闭)
    'output_dir': None,         # 输出目录，None 则使用模型所在目录
}


def parse_args() -> dict:
    """Parse command line arguments and return an evaluation config."""
    parser = argparse.ArgumentParser(description='Evaluate the seat belt classifier')
    parser.add_argument('--model-path', type=str, default=CONFIG['model_path'],
                        help='Path to a trained checkpoint. If omitted, the latest runs/train_*/best_model.pth is used.')
    parser.add_argument('--eval-dirs', nargs='+', default=CONFIG['eval_dirs'],
                        help='Evaluation directory/directories. Each can be a split root or a class folder.')
    parser.add_argument('--batch-size', type=int, default=CONFIG['batch_size'])
    parser.add_argument('--num-workers', type=int, default=CONFIG['num_workers'])
    parser.add_argument('--image-size', type=int, default=CONFIG['image_size'])
    parser.add_argument('--output-dir', type=str, default=CONFIG['output_dir'])
    parser.add_argument('--no-save-plot', action='store_true', help='Do not save confusion_matrix.png')
    parser.add_argument('--show-plot', action='store_true', help='Display the confusion matrix window')
    args = parser.parse_args()

    cfg = CONFIG.copy()
    cfg.update({
        'model_path': args.model_path,
        'eval_dirs': args.eval_dirs,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'image_size': args.image_size,
        'output_dir': args.output_dir,
        'save_plot': not args.no_save_plot,
        'show_plot': args.show_plot,
    })
    return cfg


# ==============================================================================
# CONFUSION MATRIX COMPUTATION
# ==============================================================================

class ConfusionMatrixComputer:
    """计算混淆矩阵及相关指标。"""
    
    def __init__(self, class_names: List[str] = None):
        """
        Args:
            class_names: 类别名称列表 [负类名, 正类名]
        """
        self.class_names = class_names or ['未佩戴安全带', '佩戴安全带']
        self.reset()
    
    def reset(self):
        """重置所有计数。"""
        self.predictions = []
        self.targets = []
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
        """
        更新预测结果。
        
        Args:
            preds: 模型输出的 logits (未经 sigmoid)
            targets: 真实标签
            threshold: 二分类阈值
        """
        # 转换为二分类预测
        probs = torch.sigmoid(preds)
        binary_preds = (probs >= threshold).long().view(-1)
        targets = targets.long().view(-1)
        
        self.predictions.extend(binary_preds.cpu().tolist())
        self.targets.extend(targets.cpu().tolist())
    
    def compute(self) -> Dict:
        """
        计算混淆矩阵及所有指标。
        
        Returns:
            包含混淆矩阵及各项指标的字典
        """
        preds = torch.tensor(self.predictions)
        targets = torch.tensor(self.targets)
        
        # 计算混淆矩阵的四个元素
        tp = ((preds == 1) & (targets == 1)).sum().item()
        tn = ((preds == 0) & (targets == 0)).sum().item()
        fp = ((preds == 1) & (targets == 0)).sum().item()
        fn = ((preds == 0) & (targets == 1)).sum().item()
        
        total = tp + tn + fp + fn
        
        # 计算各项指标
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # 构建混淆矩阵 (2x2)
        # 行: 真实类别, 列: 预测类别
        #           预测=0    预测=1
        # 真实=0    TN        FP
        # 真实=1    FN        TP
        confusion_matrix = np.array([
            [tn, fp],
            [fn, tp]
        ])
        
        return {
            'confusion_matrix': confusion_matrix,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'total': total,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'specificity': specificity,
            'class_names': self.class_names
        }
    
    def print_report(self, results: Dict):
        """打印详细的评估报告。"""
        cm = results['confusion_matrix']
        
        print("\n" + "=" * 60)
        print("混淆矩阵 (CONFUSION MATRIX)")
        print("=" * 60)
        
        # 打印混淆矩阵
        print("\n真实标签 \\ 预测标签")
        print("-" * 45)
        header = f"{'':20} | {results['class_names'][0]:>10} | {results['class_names'][1]:>10}"
        print(header)
        print("-" * 45)
        
        for i, row_name in enumerate(results['class_names']):
            row = f"{row_name:20} | {cm[i, 0]:>10} | {cm[i, 1]:>10}"
            print(row)
        
        print("-" * 45)
        
        # 打印详细指标
        print("\n" + "=" * 60)
        print("详细指标")
        print("=" * 60)
        print(f"\n样本统计:")
        print(f"  总样本数:   {results['total']}")
        print(f"  正类样本:   {results['tp'] + results['fn']} (佩戴安全带)")
        print(f"  负类样本:   {results['tn'] + results['fp']} (未佩戴安全带)")
        
        print(f"\n混淆矩阵元素:")
        print(f"  TP (True Positive):  {results['tp']:>6} - 正确预测佩戴安全带")
        print(f"  TN (True Negative):  {results['tn']:>6} - 正确预测未佩戴安全带")
        print(f"  FP (False Positive): {results['fp']:>6} - 错误预测为佩戴 (实际未佩戴)")
        print(f"  FN (False Negative): {results['fn']:>6} - 错误预测为未佩戴 (实际佩戴)")
        
        print(f"\n性能指标:")
        print(f"  准确率 (Accuracy):    {results['accuracy']:.4f}  = (TP+TN)/(TP+TN+FP+FN)")
        print(f"  精确率 (Precision):   {results['precision']:.4f}  = TP/(TP+FP)")
        print(f"  召回率 (Recall):      {results['recall']:.4f}  = TP/(TP+FN)")
        print(f"  F1 分数 (F1-Score):   {results['f1']:.4f}  = 2*P*R/(P+R)")
        print(f"  特异度 (Specificity): {results['specificity']:.4f}  = TN/(TN+FP)")
        
        print("\n" + "=" * 60)


def plot_confusion_matrix(results: Dict, save_path: str = None, show: bool = True):
    """
    可视化混淆矩阵。
    
    Args:
        results: compute() 返回的结果字典
        save_path: 保存图片的路径
        show: 是否显示图片
    """
    if not HAS_VISUALIZATION:
        print("跳过可视化: 未安装必要的库")
        return
    
    cm = results['confusion_matrix']
    class_names = results['class_names']
    
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. 绝对数量的混淆矩阵
    ax1 = axes[0]
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax1,
        annot_kws={'size': 16}
    )
    ax1.set_title('混淆矩阵 (数量)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('预测标签', fontsize=12)
    ax1.set_ylabel('真实标签', fontsize=12)
    
    # 2. 百分比的混淆矩阵
    ax2 = axes[1]
    cm_normalized = cm.astype('float') / cm.sum() * 100
    sns.heatmap(
        cm_normalized, 
        annot=True, 
        fmt='.1f', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax2,
        annot_kws={'size': 16}
    )
    ax2.set_title('混淆矩阵 (百分比 %)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('预测标签', fontsize=12)
    ax2.set_ylabel('真实标签', fontsize=12)
    
    # 添加总标题
    fig.suptitle(
        f'安全带检测模型评估\n'
        f'Accuracy: {results["accuracy"]:.2%} | '
        f'Precision: {results["precision"]:.2%} | '
        f'Recall: {results["recall"]:.2%} | '
        f'F1: {results["f1"]:.2%}',
        fontsize=12,
        y=1.02
    )
    
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n混淆矩阵图片已保存到: {save_path}")
    
    # 显示图片
    if show:
        plt.show()
    
    plt.close()


# ==============================================================================
# MODEL LOADING
# ==============================================================================

def find_latest_model(runs_dir: Path) -> Path:
    """查找最新的 best_model.pth。"""
    run_dirs = sorted(runs_dir.glob('train_*'), reverse=True)
    
    for run_dir in run_dirs:
        model_path = run_dir / 'best_model.pth'
        if model_path.exists():
            return model_path
    
    raise FileNotFoundError(f"在 {runs_dir} 中未找到 best_model.pth")


def load_model(model_path: Path, device: torch.device) -> nn.Module:
    """加载模型。"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # 获取模型配置
    if 'config' in checkpoint:
        cfg = checkpoint['config']
        model_config = SeatBeltDetectorConfig(
            n_patches=cfg.get('n_patches', 6),
            patch_size=(64, 64),
            patch_output_size=(224, 224),
            pretrained=False,
            freeze_backbone=False,
            feature_dim=960,
            attention_dropout=0.1,
            use_residual=True,
            use_layer_norm=True,
            gru_hidden_dim=cfg.get('gru_hidden_dim', 256),
            gru_num_layers=1,
            gru_dropout=0.1,
            classifier_hidden_dim=128,
            classifier_dropout=cfg.get('classifier_dropout', 0.5)
        )
    else:
        # 使用默认配置
        model_config = SeatBeltDetectorConfig()
    
    model = SeatBeltDetector(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


# ==============================================================================
# MAIN EVALUATION
# ==============================================================================

@torch.no_grad()
def evaluate(cfg: dict):
    """主评估函数。"""
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'=' * 60}")
    print("安全带分类算法 - 混淆矩阵评估")
    print(f"{'=' * 60}")
    print(f"设备: {device}")
    
    # 查找模型
    if cfg['model_path'] is None:
        model_path = find_latest_model(PROJECT_ROOT / 'runs')
    else:
        model_path = Path(cfg['model_path'])
    
    print(f"模型路径: {model_path}")
    
    # 确定输出目录
    if cfg['output_dir'] is None:
        output_dir = model_path.parent
    else:
        output_dir = Path(cfg['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    print("\n加载模型...")
    model = load_model(model_path, device)
    print("模型加载成功!")
    
    # 加载数据
    print("\n加载评估数据...")
    _, eval_loader = create_dataloaders(
        train_dirs=cfg['eval_dirs'],  # 使用相同目录作为 train 和 val
        val_dirs=cfg['eval_dirs'],
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        image_size=cfg['image_size'],
        augment_train=False
    )
    print(f"评估批次数: {len(eval_loader)}")
    
    # 计算混淆矩阵
    print("\n计算混淆矩阵...")
    cm_computer = ConfusionMatrixComputer()
    
    for batch in eval_loader:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        output = model(images, return_attention=False)
        logits = output['logits']
        
        cm_computer.update(logits, labels)
    
    # 获取结果
    results = cm_computer.compute()
    
    # 打印报告
    cm_computer.print_report(results)
    
    # 保存结果到 JSON
    results_json = {
        'confusion_matrix': results['confusion_matrix'].tolist(),
        'tp': results['tp'],
        'tn': results['tn'],
        'fp': results['fp'],
        'fn': results['fn'],
        'total': results['total'],
        'accuracy': results['accuracy'],
        'precision': results['precision'],
        'recall': results['recall'],
        'f1': results['f1'],
        'specificity': results['specificity'],
        'class_names': results['class_names'],
        'model_path': str(model_path)
    }
    
    json_path = output_dir / 'confusion_matrix_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {json_path}")
    
    # 可视化
    if cfg['save_plot'] or cfg['show_plot']:
        plot_path = output_dir / 'confusion_matrix.png' if cfg['save_plot'] else None
        plot_confusion_matrix(results, save_path=str(plot_path), show=cfg['show_plot'])
    
    return results


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == '__main__':
    evaluate(parse_args())
