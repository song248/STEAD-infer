"""
Sliding window frame-level inference for violence detection.
- 2 sec inference window
- 1 sec stride
- Overlapping sections averaged at frame level
- Output: frame-level CSV (frame, violence)
"""

import os
import math
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch import nn
from pytorchvideo.data.encoded_video import EncodedVideo
from model import Model

# ===== CONFIG =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "saved_models/888tiny.pkl"

video_root_folder = "hf-violence/video"     # 처리할 영상 폴더
frame_score_folder = "hf-violence/predict-ft"  # 프레임 단위 점수 CSV 저장 폴더
os.makedirs(frame_score_folder, exist_ok=True)

threshold = 0.5
valid_exts = [".mp4", ".avi"]

# ===== X3D 설정 =====
model_name = 'x3d_l'
transform_params = {
    "num_frames": 15,
    "sampling_rate": 4,
    "side_size": 320,
    "crop_size": 320,
    "frames_per_second": 30,
}
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]

# ===== Normalize 함수 =====
def normalize_video_tensor(video_tensor, mean, std):
    mean = torch.tensor(mean, device=video_tensor.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=video_tensor.device).view(1, -1, 1, 1)
    return (video_tensor - mean) / std

# ===== 고정 간격 샘플링 =====
def fixed_interval_sample(frames, interval, num_samples):
    total_needed = interval * num_samples
    if frames.shape[0] < total_needed:
        pad = frames[-1:].repeat(total_needed - frames.shape[0], 1, 1, 1)
        frames = torch.cat([frames, pad], dim=0)
    indices = torch.arange(0, total_needed, step=interval)
    return frames[indices]

# ===== 전처리 함수 =====
def preprocess_frames(frames):
    frames = frames.permute(0, 3, 1, 2) / 255.0  # (T,H,W,C)→(T,C,H,W)
    frames = normalize_video_tensor(frames, mean, std)
    frames = torch.nn.functional.interpolate(
        frames, size=(transform_params["side_size"], transform_params["side_size"]),
        mode="bilinear", align_corners=False
    )
    crop = transform_params["crop_size"]
    center = transform_params["side_size"] // 2
    frames = frames[:, :, center - crop//2:center + crop//2,
                          center - crop//2:center + crop//2]
    frames = frames.permute(1, 0, 2, 3)  # (T,C,H,W)→(C,T,H,W)
    return frames

# ===== X3D 모델 로딩 =====
x3d = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)
x3d = nn.Sequential(*list(x3d.blocks[:5]))  # classification head 제거
x3d = x3d.to(device).eval()

# ===== 이상 판단 함수 =====
def predict_anomaly(feature, model):
    with torch.no_grad():
        score, _ = model(feature.unsqueeze(0).to(device))  # (1,C,T,H,W)
        prob = torch.sigmoid(score).item()
    return prob

# ===== 프레임 단위 슬라이딩 인퍼런스 =====
def frame_level_inference(video_path, model, threshold=0.5):
    video = EncodedVideo.from_path(video_path)
    fps = transform_params["frames_per_second"]
    duration = float(video.duration)
    window_sec = (transform_params["num_frames"] * transform_params["sampling_rate"]) / fps
    stride_sec = 1.0

    num_frames = int(duration * fps)
    frame_scores = np.zeros(num_frames)
    frame_counts = np.zeros(num_frames)

    total_steps = max(1, math.ceil((duration - window_sec) / stride_sec) + 1)

    for step in tqdm(range(total_steps), desc=f"  Processing windows", leave=False, ncols=80):
        start = step * stride_sec
        end = start + window_sec
        try:
            clip = video.get_clip(start_sec=start, end_sec=end)
        except:
            break

        frames = clip["video"]
        if frames.shape[0] == 3:
            frames = frames.permute(1, 2, 3, 0)
        elif frames.shape[-1] != 3:
            raise ValueError(f"Unsupported frame shape: {frames.shape}")

        frames = fixed_interval_sample(frames, transform_params["sampling_rate"], transform_params["num_frames"])
        frames = preprocess_frames(frames).unsqueeze(0).to(device)

        with torch.no_grad():
            feature = x3d(frames).squeeze(0)
            score = predict_anomaly(feature, model)

        # 프레임 단위 누적
        start_frame = int(start * fps)
        end_frame = min(int(end * fps), num_frames)
        frame_scores[start_frame:end_frame] += score
        frame_counts[start_frame:end_frame] += 1

    # 평균 확률 → 이상 여부 변환
    frame_scores /= np.maximum(frame_counts, 1)
    frame_labels = (frame_scores >= threshold).astype(int)
    return frame_labels

# ===== MAIN =====
if __name__ == "__main__":
    model = Model(ff_mult=1, dims=(32, 32), depths=(1, 1)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    video_files = [
        fname for fname in os.listdir(video_root_folder)
        if os.path.splitext(fname)[-1].lower() in valid_exts
    ]

    for idx, fname in enumerate(video_files, start=1):
        print(f"\n[{idx}/{len(video_files)}] Processing {fname}")
        video_path = os.path.join(video_root_folder, fname)

        try:
            frame_labels = frame_level_inference(video_path, model, threshold=threshold)

            # ===== 프레임 단위 CSV 저장 =====
            video_name = os.path.splitext(fname)[0]
            save_path = os.path.join(frame_score_folder, f"{video_name}.csv")
            pd.DataFrame({
                "frame": np.arange(len(frame_labels)),
                "violence": frame_labels
            }).to_csv(save_path, index=False)
            print(f"  ✅ Saved: {save_path}")

        except Exception as e:
            print(f"  🚨 Error processing {fname}: {e}")
            continue

    print(f"\n✅ All frame-level CSVs saved in folder: {frame_score_folder}")
