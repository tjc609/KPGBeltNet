# Seat Belt Detection using Part-to-Whole Attention

This repository contains the core implementation for the paper:

**"Seat Belt Detection using Part-to-Whole Attention on Diagonally Sampled Patches"**

The repository is intentionally kept minimal so that it can be cloned and redeployed on another machine. It includes model modules, data loading code, and training/evaluation/inference scripts. Training outputs, checkpoints, videos, full datasets, and local experiment drafts are excluded from Git.

## Method Overview

The model classifies whether a cropped upper-body ROI contains a visible seat belt.

```text
Input ROI image
    -> diagonal patch sampler
    -> global MobileNetV3 + local MobileNetV3
    -> Part-to-Whole attention
    -> Bi-GRU sequence encoder
    -> binary classifier
```

## Repository Structure

```text
SeatBeltV1/
├── data/
│   ├── __init__.py
│   └── dataset.py
├── models/
│   ├── __init__.py
│   ├── attention.py
│   ├── classifier.py
│   ├── feature_extractor.py
│   ├── patch_sampler.py
│   ├── pipeline.py
│   └── sequence_encoder.py
├── scripts/
│   ├── train.py
│   ├── inference.py
│   ├── evaluate_confusion_matrix.py
│   └── detect_seatbelt.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Python 3.9+ is recommended.

```bash
git clone https://github.com/YOUR_NAME/SeatBeltV1.git
cd SeatBeltV1

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For CUDA training, install the PyTorch build that matches your CUDA version before installing the remaining dependencies.

## Dataset

The paper dataset is not included in this repository.

**Download link will be available upon publication of our paper.**

After downloading, place the dataset in an ImageFolder-style class directory layout:

```text
data/
├── train/
│   ├── PassengerWithSeatBelt/
│   │   ├── image001.jpg
│   │   └── ...
│   └── PassengerWithoutSeatBelt/
│       ├── image001.jpg
│       └── ...
└── val/
    ├── PassengerWithSeatBelt/
    │   ├── image001.jpg
    │   └── ...
    └── PassengerWithoutSeatBelt/
        ├── image001.jpg
        └── ...
```

Accepted positive class folder names include `PassengerWithSeatBelt`, `PassengerWithSeatBelt_aug`, `WithSeatBelt`, `with_belt`, `positive`, and `1`.

Accepted negative class folder names include `PassengerWithoutSeatBelt`, `PassengerWithoutSeatBelt_aug`, `WithoutSeatBelt`, `without_belt`, `negative`, and `0`.

Full datasets are ignored by Git through `.gitignore`.

## Training

```bash
python scripts/train.py \
  --train-dirs data/train \
  --val-dirs data/val \
  --epochs 50 \
  --batch-size 64 \
  --lr 1e-4
```

By default, checkpoints and logs are saved under `runs/train_YYYYmmdd_HHMMSS/`.

Useful options:

```bash
python scripts/train.py --help
```

## Evaluation

Evaluate a trained checkpoint on the validation set:

```bash
python scripts/evaluate_confusion_matrix.py \
  --model-path runs/train_YYYYmmdd_HHMMSS/best_model.pth \
  --eval-dirs data/val
```

This saves `confusion_matrix_results.json` and, when visualization dependencies are installed, `confusion_matrix.png`.

## Classifier Inference

Run the classifier on cropped upper-body ROI images:

```bash
python scripts/inference.py \
  --checkpoint runs/train_YYYYmmdd_HHMMSS/best_model.pth \
  --image-dir data/test \
  --output-dir runs/inference
```

For a single image:

```bash
python scripts/inference.py \
  --checkpoint runs/train_YYYYmmdd_HHMMSS/best_model.pth \
  --image path/to/roi.jpg
```

## Full Detection Pipeline

`scripts/detect_seatbelt.py` combines a YOLO pose model with the trained seat belt classifier. You must provide both weights:

```bash
python scripts/detect_seatbelt.py \
  --source path/to/image_or_video \
  --yolo-weights path/to/yolo-pose.pt \
  --seatbelt-weights runs/train_YYYYmmdd_HHMMSS/best_model.pth
```

YOLO pose weights and trained seat belt checkpoints are not included in this repository.

## Notes

- Do not commit `runs/`, datasets, model weights, videos, or generated inference outputs.
- Publish trained checkpoints through GitHub Releases, cloud storage, or Git LFS if needed.
- The repository tracks only the source files required to retrain and redeploy the method.

## Citation

If you use this code, please cite:

```bibtex
@article{seatbelt2024,
  title={Seat Belt Detection using Part-to-Whole Attention on Diagonally Sampled Patches},
  author={...},
  journal={...},
  year={2024}
}
```
