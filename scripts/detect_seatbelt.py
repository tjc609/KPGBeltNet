#!/usr/bin/env python3
# Seat Belt Detection Pipeline
# Author: Research Team
# License: MIT

"""
Run seat belt detection inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python scripts/detect_seatbelt.py --source img.jpg                 # image
    $ python scripts/detect_seatbelt.py --source vid.mp4                 # video
    $ python scripts/detect_seatbelt.py --source path/                   # directory
    $ python scripts/detect_seatbelt.py --source 'path/*.jpg'            # glob
    $ python scripts/detect_seatbelt.py --source 0                       # webcam
    $ python scripts/detect_seatbelt.py --source screen                  # screenshot
    $ python scripts/detect_seatbelt.py --source 'rtsp://example.com/media.mp4'  # RTSP stream

Usage - options:
    $ python scripts/detect_seatbelt.py --source img.jpg --view-img      # show results
    $ python scripts/detect_seatbelt.py --source vid.mp4 --save-crop     # save ROI crops
    $ python scripts/detect_seatbelt.py --source path/ --nosave          # don't save results
"""

import argparse
import csv
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # SeatBeltV1 root directory
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import SeatBeltDetector, SeatBeltDetectorConfig

# Supported formats
IMG_FORMATS = {'bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'webp', 'pfm'}
VID_FORMATS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm', 'gif', 'm4v'}


def colorstr(*args):
    """Colors a string for terminal output."""
    colors = {
        'blue': '\033[34m', 'cyan': '\033[36m', 'green': '\033[32m',
        'red': '\033[31m', 'yellow': '\033[33m', 'bold': '\033[1m',
        'end': '\033[0m'
    }
    *prefix, string = args
    colored = ''.join(colors.get(p, '') for p in prefix) + str(string) + colors['end']
    return colored


def print_args(args: dict):
    """Print function arguments."""
    print(colorstr('bold', 'Arguments: ') + ', '.join(f'{k}={v}' for k, v in args.items()))


def increment_path(path: Path, exist_ok: bool = False, sep: str = '', mkdir: bool = False):
    """Increment file or directory path, i.e. runs/exp --> runs/exp2."""
    path = Path(path)
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'
            if not Path(p).exists():
                path = Path(p)
                break
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    return path


class Profile:
    """Profile class for timing code execution."""
    def __init__(self):
        self.t = 0.0
        self.dt = 0.0
    
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.dt = time.perf_counter() - self.start
        self.t += self.dt


# ==============================================================================
# DATA LOADERS
# ==============================================================================

class LoadImages:
    """Image/video dataloader for inference."""
    
    def __init__(self, path: str, vid_stride: int = 1):
        path = str(Path(path).resolve())
        
        if '*' in path:
            files = sorted(Path().glob(path))
        elif Path(path).is_dir():
            files = sorted(Path(path).iterdir())
        elif Path(path).is_file():
            files = [Path(path)]
        else:
            raise FileNotFoundError(f'Invalid path: {path}')
        
        images = [f for f in files if f.suffix[1:].lower() in IMG_FORMATS]
        videos = [f for f in files if f.suffix[1:].lower() in VID_FORMATS]
        
        self.files = images + videos
        self.nf = len(self.files)
        self.video_flag = [False] * len(images) + [True] * len(videos)
        self.mode = 'image'
        self.vid_stride = vid_stride
        self.cap = None
        self.frame = 0
        self.frames = 0
        
        if len(videos) > 0:
            self._new_video(videos[0])
        
        assert self.nf > 0, f'No images or videos found in {path}'
    
    def __iter__(self):
        self.count = 0
        return self
    
    def __next__(self):
        if self.count >= self.nf:
            raise StopIteration
        
        path = self.files[self.count]
        
        if self.video_flag[self.count]:
            self.mode = 'video'
            for _ in range(self.vid_stride):
                ret, im0 = self.cap.read()
                self.frame += 1
                if not ret:
                    self.cap.release()
                    self.count += 1
                    if self.count < self.nf:
                        self._new_video(self.files[self.count])
                    return self.__next__()
            s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '
        else:
            self.mode = 'image'
            self.count += 1
            im0 = cv2.imread(str(path))
            assert im0 is not None, f'Image not found: {path}'
            s = f'image {self.count}/{self.nf} {path}: '
        
        return str(path), im0, self.cap, s
    
    def _new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(str(path))
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)
    
    def __len__(self):
        return self.nf


class LoadStreams:
    """Stream dataloader for webcam/RTSP/HTTP streams."""
    
    def __init__(self, source: str = '0', vid_stride: int = 1):
        self.mode = 'stream'
        self.vid_stride = vid_stride
        
        source = int(source) if source.isnumeric() else source
        self.cap = cv2.VideoCapture(source)
        
        assert self.cap.isOpened(), f'Failed to open stream: {source}'
        
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.frame = 0
        
        # Read first frame
        ret, self.img = self.cap.read()
        assert ret, 'Failed to read from stream'
        
        print(f'Stream opened: {source} ({self.w}x{self.h} @ {self.fps:.1f}fps)')
    
    def __iter__(self):
        self.count = -1
        return self
    
    def __next__(self):
        self.count += 1
        
        for _ in range(self.vid_stride):
            ret, im0 = self.cap.read()
            if not ret:
                self.cap.release()
                cv2.destroyAllWindows()
                raise StopIteration
            self.frame += 1
        
        s = f'stream frame {self.frame}: '
        return str(self.count), im0, None, s
    
    def __len__(self):
        return 999999


class LoadScreenshots:
    """Screenshot dataloader."""
    
    def __init__(self):
        try:
            import mss
            self.sct = mss.mss()
            self.monitor = self.sct.monitors[1]
            self.mode = 'screen'
            self.frame = 0
            print(f'Screen capture: {self.monitor["width"]}x{self.monitor["height"]}')
        except ImportError:
            raise ImportError("Screen capture requires 'mss': pip install mss")
    
    def __iter__(self):
        return self
    
    def __next__(self):
        im0 = np.array(self.sct.grab(self.monitor))
        im0 = cv2.cvtColor(im0, cv2.COLOR_BGRA2BGR)
        self.frame += 1
        s = f'screen {self.frame}: '
        return 'screen', im0, None, s
    
    def __len__(self):
        return 999999


# ==============================================================================
# MODELS
# ==============================================================================

class YOLOPoseDetector:
    """YOLO Pose model wrapper for person detection."""
    
    def __init__(self, weights: str, conf_thres: float = 0.5, device: str = ''):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf_thres = conf_thres
        self.device = device
    
    def __call__(
        self,
        im0: np.ndarray,
        keypoint_indices: List[int] = [5, 6, 11, 12],
        padding: int = 15,
        max_det: int = 10
    ) -> List[Dict]:
        """
        Detect persons and extract upper body ROIs.
        
        Returns:
            List of dicts with 'roi', 'bbox', 'keypoints', 'area', 'person_idx'
        """
        h, w = im0.shape[:2]
        results = self.model.predict(im0, verbose=False, classes=[0], device=self.device)
        
        if not results or results[0].keypoints is None:
            return []
        
        kpts = results[0].keypoints.data.cpu().numpy()
        detections = []
        
        for idx, person_kpts in enumerate(kpts):
            valid_pts = []
            for k in keypoint_indices:
                x, y, c = person_kpts[k]
                if c > self.conf_thres and x > 1 and y > 1:
                    valid_pts.append([x, y])
            
            if len(valid_pts) < 2:
                continue
            
            pts = np.array(valid_pts)
            x1 = max(0, int(pts[:, 0].min() - padding))
            y1 = max(0, int(pts[:, 1].min() - padding))
            x2 = min(w, int(pts[:, 0].max() + padding))
            y2 = min(h, int(pts[:, 1].max() + padding))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            detections.append({
                'roi': im0[y1:y2, x1:x2].copy(),
                'bbox': [x1, y1, x2, y2],
                'keypoints': person_kpts,
                'area': (x2 - x1) * (y2 - y1),
                'person_idx': idx
            })
        
        # Sort by area descending
        detections.sort(key=lambda x: x['area'], reverse=True)
        return detections[:max_det]


class SeatBeltClassifierWrapper:
    """Seat belt classification model wrapper."""
    
    def __init__(self, weights: str, device: torch.device):
        self.device = device
        checkpoint = torch.load(weights, map_location=device)
        
        cfg = checkpoint.get('config', {})
        model_cfg = SeatBeltDetectorConfig(
            n_patches=cfg.get('n_patches', 5),
            gru_hidden_dim=cfg.get('gru_hidden_dim', 256),
            classifier_dropout=cfg.get('classifier_dropout', 0.5),
            pretrained=False
        )
        
        self.model = SeatBeltDetector(model_cfg)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device).eval()
        
        import torchvision.transforms as T
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        f1 = checkpoint.get('best_f1', 0)
        print(f'Seat belt model loaded: {weights} (F1={f1:.4f})')
    
    @torch.no_grad()
    def __call__(self, roi: np.ndarray) -> Dict:
        """Classify seat belt status from ROI image."""
        img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        x = self.transform(img).unsqueeze(0).to(self.device)
        
        out = self.model(x, return_attention=True)
        prob = out['probabilities'].item()
        pred = int(prob >= 0.5)
        
        return {
            'prediction': pred,
            'probability': prob,
            'confidence': prob if pred else 1 - prob,
            'label': 'WITH_SEATBELT' if pred else 'NO_SEATBELT'
        }


# ==============================================================================
# VISUALIZATION
# ==============================================================================

SKELETON = [(5, 6), (5, 11), (6, 12), (11, 12)]
KPT_COLORS = {5: (255, 0, 0), 6: (255, 0, 0), 11: (0, 255, 255), 12: (0, 255, 255)}


class Annotator:
    """Image annotator for drawing boxes, labels, and keypoints."""
    
    def __init__(self, im: np.ndarray, line_width: int = 2):
        self.im = im.copy()
        self.lw = line_width
    
    def box_label(self, box, label: str = '', color=(0, 255, 0)):
        """Draw box with label."""
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(self.im, (x1, y1), (x2, y2), color, self.lw)
        
        if label:
            tf = max(self.lw - 1, 1)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.lw / 3, tf)
            cv2.rectangle(self.im, (x1, y1 - th - 3), (x1 + tw, y1), color, -1)
            cv2.putText(self.im, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX,
                       self.lw / 3, (255, 255, 255), tf, cv2.LINE_AA)
    
    def keypoints(self, kpts, kpt_indices: List[int], conf_thres: float = 0.5, color=(0, 255, 0)):
        """Draw keypoints and skeleton."""
        # Skeleton
        for i, j in SKELETON:
            if kpts[i][2] > conf_thres and kpts[j][2] > conf_thres:
                pt1 = (int(kpts[i][0]), int(kpts[i][1]))
                pt2 = (int(kpts[j][0]), int(kpts[j][1]))
                cv2.line(self.im, pt1, pt2, color, self.lw)
        
        # Points
        for k in kpt_indices:
            if kpts[k][2] > conf_thres:
                pt = (int(kpts[k][0]), int(kpts[k][1]))
                cv2.circle(self.im, pt, 5, KPT_COLORS.get(k, color), -1)
                cv2.circle(self.im, pt, 7, (255, 255, 255), 1)
    
    def result(self):
        return self.im


# ==============================================================================
# MAIN RUN FUNCTION
# ==============================================================================

def run(
    source: str = ROOT / 'data/images',
    yolo_weights: str = 'yolo11s-pose.pt',
    seatbelt_weights: str = ROOT / 'runs/train/best_model.pth',
    conf_thres: float = 0.5,
    device: str = '',
    view_img: bool = False,
    save_txt: bool = False,
    save_csv: bool = False,
    save_crop: bool = False,
    nosave: bool = False,
    project: str = ROOT / 'runs/detect',
    name: str = 'exp',
    exist_ok: bool = False,
    line_thickness: int = 2,
    vid_stride: int = 1,
    padding: int = 15,
    max_det: int = 10,
    keypoints: List[int] = [5, 6, 11, 12],
):
    """
    Run seat belt detection inference.
    
    Args:
        source: Input source (file/dir/URL/webcam/screen)
        yolo_weights: Path to YOLO pose weights
        seatbelt_weights: Path to seat belt classifier weights
        conf_thres: Confidence threshold
        device: CUDA device ('', '0', 'cpu')
        view_img: Show results
        save_txt: Save results to txt
        save_csv: Save results to CSV
        save_crop: Save cropped ROIs
        nosave: Don't save images
        project: Save results to project/name
        name: Save results to project/name
        exist_ok: Existing project/name ok
        line_thickness: Bounding box thickness
        vid_stride: Video frame-rate stride
        padding: ROI padding pixels
        max_det: Maximum detections per frame
        keypoints: Keypoint indices for ROI
    """
    source = str(source)
    save_img = not nosave
    is_file = Path(source).suffix[1:].lower() in (IMG_FORMATS | VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower() == 'screen'
    
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    (save_dir / 'crops').mkdir(parents=True, exist_ok=True) if save_crop else None
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device(f'cuda:{device}' if device.isnumeric() else 
                         'cuda' if torch.cuda.is_available() and device != 'cpu' else 'cpu')
    
    # Load models
    print(colorstr('bold', 'Loading models...'))
    yolo = YOLOPoseDetector(yolo_weights, conf_thres, str(device))
    classifier = SeatBeltClassifierWrapper(str(seatbelt_weights), device)
    
    # Dataloader
    if webcam:
        dataset = LoadStreams(source, vid_stride)
    elif screenshot:
        dataset = LoadScreenshots()
    else:
        dataset = LoadImages(source, vid_stride)
    
    vid_path, vid_writer = None, None
    
    # Run inference
    seen = 0
    dt = (Profile(), Profile(), Profile())  # profiling: detect, classify, total
    results_data = []
    stats = {'with_seatbelt': 0, 'no_seatbelt': 0, 'no_person': 0}
    
    # CSV setup
    if save_csv:
        csv_path = save_dir / 'results.csv'
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['frame', 'person_idx', 'label', 'confidence', 'x1', 'y1', 'x2', 'y2'])
    
    print(colorstr('bold', f'\nRunning inference on {source}...'))
    
    for path, im0, vid_cap, s in dataset:
        seen += 1
        
        with dt[0]:  # Person detection
            persons = yolo(im0, keypoints, padding, max_det)
        
        if not persons:
            stats['no_person'] += 1
            print(f'{s}no persons detected')
            continue
        
        annotator = Annotator(im0, line_thickness)
        frame_results = []
        
        with dt[1]:  # Classification
            for p in persons:
                result = classifier(p['roi'])
                p.update(result)
                
                # Stats
                if result['prediction']:
                    stats['with_seatbelt'] += 1
                else:
                    stats['no_seatbelt'] += 1
                
                # Annotate
                color = (0, 255, 0) if result['prediction'] else (0, 0, 255)
                label = f"P{p['person_idx']}: {result['label']} {result['confidence']:.0%}"
                annotator.box_label(p['bbox'], label, color)
                annotator.keypoints(p['keypoints'], keypoints, conf_thres, color)
                
                frame_results.append({
                    'person_idx': p['person_idx'],
                    'bbox': p['bbox'],
                    'label': result['label'],
                    'confidence': result['confidence']
                })
                
                # Save crop
                if save_crop:
                    crop_path = save_dir / 'crops' / f'{Path(path).stem}_p{p["person_idx"]}_{result["label"]}.jpg'
                    cv2.imwrite(str(crop_path), p['roi'])
                
                # CSV
                if save_csv:
                    csv_writer.writerow([
                        Path(path).stem, p['person_idx'], result['label'],
                        f"{result['confidence']:.4f}", *p['bbox']
                    ])
        
        im0 = annotator.result()
        
        # Print results
        n_with = sum(1 for r in frame_results if 'WITH' in r['label'])
        n_without = len(frame_results) - n_with
        print(f"{s}{len(frame_results)} persons ({n_with} with, {n_without} without seatbelt), "
              f"{dt[0].dt * 1e3:.1f}ms detect, {dt[1].dt * 1e3:.1f}ms classify")
        
        # Store results
        results_data.append({'source': path, 'detections': frame_results})
        
        # Save txt
        if save_txt:
            txt_path = save_dir / 'labels' / f'{Path(path).stem}.txt'
            txt_path.parent.mkdir(exist_ok=True)
            with open(txt_path, 'a') as f:
                for r in frame_results:
                    f.write(f"{r['person_idx']} {r['label']} {r['confidence']:.4f} "
                           f"{' '.join(map(str, r['bbox']))}\n")
        
        # Display
        if view_img:
            if platform.system() == 'Linux':
                cv2.namedWindow('Seat Belt Detection', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.imshow('Seat Belt Detection', im0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Save image/video
        if save_img:
            if dataset.mode == 'image':
                cv2.imwrite(str(save_dir / Path(path).name), im0)
            else:
                if vid_path != path:
                    vid_path = path
                    if vid_writer:
                        vid_writer.release()
                    if vid_cap:
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(save_dir / f'{Path(path).stem}.mp4')
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)
    
    # Cleanup
    if vid_writer:
        vid_writer.release()
    if save_csv:
        csv_file.close()
    cv2.destroyAllWindows()
    
    # Save JSON results
    json_path = save_dir / 'results.json'
    with open(json_path, 'w') as f:
        json.dump({'stats': stats, 'results': results_data}, f, indent=2)
    
    # Print summary
    t = sum(p.t for p in dt)
    print(f'\n{colorstr("bold", "Results saved to")} {colorstr("blue", save_dir)}')
    print(f'Speed: {t / seen * 1e3:.1f}ms per frame ({seen} frames)')
    print(f'Stats: {stats["with_seatbelt"]} with seatbelt, {stats["no_seatbelt"]} without, '
          f'{stats["no_person"]} no person')


def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Seat Belt Detection Inference')
    parser.add_argument('--source', type=str, required=True, help='file/dir/URL/webcam/screen')
    parser.add_argument('--yolo-weights', type=str, required=True, help='YOLO pose model path')
    parser.add_argument('--seatbelt-weights', type=str, required=True, help='Seat belt classifier weights')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--view-img', action='store_true', help='Show results')
    parser.add_argument('--save-txt', action='store_true', help='Save results to txt')
    parser.add_argument('--save-csv', action='store_true', help='Save results to CSV')
    parser.add_argument('--save-crop', action='store_true', help='Save cropped ROIs')
    parser.add_argument('--nosave', action='store_true', help="Don't save images/videos")
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='Save to project/name')
    parser.add_argument('--name', default='exp', help='Save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='Existing project/name ok')
    parser.add_argument('--line-thickness', type=int, default=2, help='Box thickness')
    parser.add_argument('--vid-stride', type=int, default=1, help='Video frame-rate stride')
    parser.add_argument('--padding', type=int, default=15, help='ROI padding')
    parser.add_argument('--max-det', type=int, default=10, help='Max detections per frame')
    
    return parser.parse_args()


def main(opt):
    """Main function."""
    print_args(vars(opt))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
