#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Precomputed feature (.npy) based frame-level inference for violence detection.
- Loads features from: hf_npy/{video_name}.npy
- Assumes features are per 2s window with 1s stride (same as original pipeline)
- Output: frame-level CSV (frame, prob, violence)
"""

import os
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from model import Model
from pytorchvideo.data.encoded_video import EncodedVideo  # duration/fps only

# ===== DEFAULT CONFIG =====
DEFAULT_MODEL_PATH = "saved_models/ft_model.pkl"
DEFAULT_VIDEO_ROOT = "hf-violence/video"
DEFAULT_PRED_ROOT = "hf-violence/predict-ft"
DEFAULT_FEAT_ROOT = "hf_npy"
DEFAULT_THRESHOLD = 0.5
VALID_EXTS = (".mp4", ".avi")

# Original pipeline timing params
TRANSFORM_PARAMS = {
    "num_frames": 15,
    "sampling_rate": 4,
    "side_size": 320,
    "crop_size": 320,
    "frames_per_second": 30,
}

def get_window_stride_seconds():
    fps = TRANSFORM_PARAMS["frames_per_second"]
    window_sec = (TRANSFORM_PARAMS["num_frames"] * TRANSFORM_PARAMS["sampling_rate"]) / fps
    stride_sec = 1.0  # original: 2s window, 1s stride
    return window_sec, stride_sec, fps

@torch.no_grad()
def predict_anomaly(feature_4d: torch.Tensor, model: torch.nn.Module, device: torch.device) -> float:
    """
    feature_4d: (C, T, H, W)
    model: returns (score, aux)
    """
    score, _ = model(feature_4d.unsqueeze(0).to(device))  # (1,C,T,H,W)
    return torch.sigmoid(score).item()

def frame_level_inference_from_npy(video_path: str,
                                   feature_npy_path: str,
                                   model: torch.nn.Module,
                                   device: torch.device) -> np.ndarray:
    """Return per-frame probability (float32, 0~1)."""
    # Use duration to set number of frames; no decoding
    video = EncodedVideo.from_path(video_path)
    duration = float(video.duration)

    window_sec, stride_sec, fps = get_window_stride_seconds()

    num_frames = int(duration * fps)
    frame_scores = np.zeros(num_frames, dtype=np.float32)
    frame_counts = np.zeros(num_frames, dtype=np.float32)

    # Load precomputed features
    npy = np.load(feature_npy_path, allow_pickle=False)
    # Expected shapes: (N, C, T, H, W) or (C, T, H, W)
    if npy.ndim == 4:
        npy = npy[None, ...]  # (1, C, T, H, W)
    if npy.ndim != 5:
        raise ValueError(f"Expected 4D/5D feature, got shape {npy.shape}")

    total_steps = npy.shape[0]

    for step in tqdm(range(total_steps), desc="  Processing feature windows", leave=False, ncols=80):
        feature = torch.from_numpy(npy[step]).float().to(device)  # (C,T,H,W)
        score = predict_anomaly(feature, model, device)  # sigmoid probability

        start = step * stride_sec
        end = start + window_sec

        # Accumulate window score onto covered frames
        start_frame = int(start * fps)
        end_frame = min(int(end * fps), num_frames)
        if start_frame < end_frame:
            frame_scores[start_frame:end_frame] += score
            frame_counts[start_frame:end_frame] += 1

    # Average probability per frame
    probs = frame_scores / np.maximum(frame_counts, 1.0)
    return probs.astype(np.float32)

def main():
    parser = argparse.ArgumentParser(description="NPY-based frame-level inference (saves prob + binary).")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--video_root", default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--feat_root",  default=DEFAULT_FEAT_ROOT)
    parser.add_argument("--out_root",   default=DEFAULT_PRED_ROOT)
    parser.add_argument("--threshold",  type=float, default=DEFAULT_THRESHOLD,
                        help="Optional default threshold used to also save a binary column for compatibility.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    device = torch.device(args.device)

    # Load model
    model = Model(ff_mult=1, dims=(32, 32), depths=(1, 1)).to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Enumerate videos
    video_files = [
        f for f in os.listdir(args.video_root)
        if os.path.splitext(f)[-1].lower() in VALID_EXTS
    ]
    video_files.sort()

    for idx, fname in enumerate(video_files, start=1):
        print(f"\n[{idx}/{len(video_files)}] Processing {fname}")
        video_path = os.path.join(args.video_root, fname)
        vid = os.path.splitext(fname)[0]
        feat_path = os.path.join(args.feat_root, f"{vid}.npy")

        if not os.path.isfile(feat_path):
            print(f"  ⚠️ Feature not found: {feat_path} (skip)")
            continue

        try:
            probs = frame_level_inference_from_npy(video_path, feat_path, model, device)
            # Also provide a binary column for backward compatibility (optional)
            violence = (probs >= args.threshold).astype(np.int32)

            save_path = os.path.join(args.out_root, f"{vid}.csv")
            df = pd.DataFrame({
                "frame": np.arange(len(probs), dtype=np.int32),
                "prob": probs,
                "violence": violence,
            })
            df.to_csv(save_path, index=False)
            print(f"  ✅ Saved: {save_path}  (columns: frame, prob, violence)")
        except Exception as e:
            print(f"  🚨 Error processing {fname}: {e}")
            continue

    print(f"\n✅ All frame-level CSVs saved in folder: {args.out_root}")

if __name__ == "__main__":
    main()
