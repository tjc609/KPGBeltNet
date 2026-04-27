# KPGBeltNet: In-Vehicle Seatbelt Detection Based on Keypoint-Guided Sampling and Local-Global Attention

This repository provides the core implementation of **KPGBeltNet**, an in-vehicle seatbelt detection method designed for cropped upper-body regions of vehicle occupants.

KPGBeltNet uses human keypoint-guided diagonal patch sampling and a local-global attention mechanism to enhance discriminative seatbelt features under in-vehicle occlusion, pose variation, and illumination changes.

The proposed framework comprises three core components:

1. Keypoint-guided ROI and diagonal patch sampling.
2. Dual-stream MobileNetV3 feature extraction for global and local visual cues.
3. Local-global attention with a Bi-GRU sequence encoder for binary seatbelt classification.

---

## Requirements

Python 3.9+ is recommended. The main dependencies are PyTorch, torchvision, OpenCV, Ultralytics, NumPy, Pillow, matplotlib, and seaborn.

Install the environment as follows:

```bash
git clone https://github.com/tjc609/KPGBeltNet.git
cd KPGBeltNet

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the virtual environment with:

```bash
.venv\Scripts\activate
```

For GPU training, install the PyTorch build that matches your CUDA version before installing the remaining packages.

---

## Dataset

The dataset used in the paper is not included in this repository.

**Download link will be available upon publication of our paper.**

---

## Repository Structure

```text
KPGBeltNet/
|-- data/
|   |-- __init__.py
|   `-- dataset.py
|-- models/
|   |-- __init__.py
|   |-- attention.py
|   |-- classifier.py
|   |-- feature_extractor.py
|   |-- patch_sampler.py
|   |-- pipeline.py
|   `-- sequence_encoder.py
|-- scripts/
|   |-- train.py
|   |-- inference.py
|   |-- evaluate_confusion_matrix.py
|   `-- detect_seatbelt.py
|-- requirements.txt
`-- README.md
```

---

## Train

1. Prepare the training and validation data paths.

2. Launch training:

```bash
python scripts/train.py \
  --train-dirs data/train \
  --val-dirs data/val \
  --epochs 50 \
  --batch-size 64 \
  --lr 1e-4
```

3. Resume or customize training by setting command-line options:

```bash
python scripts/train.py --help
```

Checkpoints and logs are saved to:

```text
runs/train_YYYYmmdd_HHMMSS/
```

---

## Evaluate

Evaluate a trained checkpoint on the validation set:

```bash
python scripts/evaluate_confusion_matrix.py \
  --model-path runs/train_YYYYmmdd_HHMMSS/best_model.pth \
  --eval-dirs data/val
```

The script reports accuracy, precision, recall, F1 score, specificity, and the confusion matrix. It also saves evaluation results under the checkpoint directory unless another output directory is specified.

---

## Inference

Run classifier inference on cropped upper-body ROI images:

```bash
python scripts/inference.py \
  --checkpoint runs/train_YYYYmmdd_HHMMSS/best_model.pth \
  --image-dir data/test \
  --output-dir runs/inference
```

For a single ROI image:

```bash
python scripts/inference.py \
  --checkpoint runs/train_YYYYmmdd_HHMMSS/best_model.pth \
  --image path/to/roi.jpg
```

---

## Full Detection Pipeline

The full detection script combines a YOLO pose model with the trained KPGBeltNet classifier. Provide both the YOLO pose weights and the trained seatbelt checkpoint:

```bash
python scripts/detect_seatbelt.py \
  --source path/to/image_or_video \
  --yolo-weights path/to/yolo-pose.pt \
  --seatbelt-weights runs/train_YYYYmmdd_HHMMSS/best_model.pth
```

YOLO pose weights and trained KPGBeltNet checkpoints are not included in this repository.

---

## Notes

- This repository only keeps the source code needed to retrain and redeploy KPGBeltNet.
- Do not commit `runs/`, datasets, model weights, videos, or generated inference outputs.
- If trained weights need to be released, use GitHub Releases, cloud storage, or Git LFS.
