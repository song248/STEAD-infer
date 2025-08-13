"""
Precomputed feature (.npy) based frame-level inference for violence detection.
- Looks for features in: hf_npy/{video_name}.npy
- Assumes features are per 2s window with 1s stride (same as original pipeline)
- Output: frame-level CSV (frame, violence)
"""

import os
import math
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from model import Model
from pytorchvideo.data.encoded_video import EncodedVideo  # duration/fps만 조회

# ===== CONFIG =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "saved_models/888tiny_finetuned_no_sw_llava_data.pkl"

video_root_folder = "hf-violence/video"     # 처리할 영상 폴더 (duration/fps 조회용)
frame_score_folder = "hf-violence/predict-ft-llava"  # 프레임 단위 점수 CSV 저장 폴더
feat_root_folder = "hf_npy"                  # 미리 추출해둔 특징 폴더
os.makedirs(frame_score_folder, exist_ok=True)

threshold = 0.5
valid_exts = [".mp4", ".avi"]

# ===== 원래 파이프라인의 시간 파라미터 재사용 =====
transform_params = {
    "num_frames": 15,
    "sampling_rate": 4,
    "side_size": 320,
    "crop_size": 320,
    "frames_per_second": 30,
}
# 2초 윈도우, 1초 스트라이드(원 코드와 동일)
def get_window_stride_seconds():
    fps = transform_params["frames_per_second"]
    window_sec = (transform_params["num_frames"] * transform_params["sampling_rate"]) / fps
    stride_sec = 1.0
    return window_sec, stride_sec, fps

# ===== 이상 판단 함수 (그대로 사용) =====
@torch.no_grad()
def predict_anomaly(feature_4d, model):
    """
    feature_4d: torch.Tensor, shape (C,T,H,W)
    model: returns (score, aux)
    """
    score, _ = model(feature_4d.unsqueeze(0).to(device))  # (1,C,T,H,W)
    return torch.sigmoid(score).item()

# ===== NPY 기반 프레임 레벨 인퍼런스 =====
def frame_level_inference_from_npy(video_path, feature_npy_path, model, threshold=0.5):
    # 비디오 길이/프레임수 파악(프레임 라벨 길이 산출용) - 디코딩/전처리는 하지 않음
    video = EncodedVideo.from_path(video_path)
    duration = float(video.duration)

    window_sec, stride_sec, fps = get_window_stride_seconds()

    num_frames = int(duration * fps)
    frame_scores = np.zeros(num_frames, dtype=np.float32)
    frame_counts = np.zeros(num_frames, dtype=np.float32)

    # 저장된 특징 불러오기
    npy = np.load(feature_npy_path, allow_pickle=False)
    # 기대 형태: (N, C, T, H, W) 또는 (C, T, H, W) 1개 윈도우
    if npy.ndim == 4:
        npy = npy[None, ...]  # (1, C, T, H, W)
    assert npy.ndim == 5, f"Expected 4D/5D feature, got shape {npy.shape}"

    total_steps = npy.shape[0]

    for step in tqdm(range(total_steps), desc="  Processing feature windows", leave=False, ncols=80):
        feature = torch.from_numpy(npy[step]).float().to(device)  # (C,T,H,W)
        score = predict_anomaly(feature, model)

        start = step * stride_sec
        end = start + window_sec

        # 프레임 단위 누적(원 코드와 동일 로직)
        start_frame = int(start * fps)
        end_frame = min(int(end * fps), num_frames)
        if start_frame < end_frame:  # 안전장치
            frame_scores[start_frame:end_frame] += score
            frame_counts[start_frame:end_frame] += 1

    # 평균 확률 → 이상 여부 변환(원 코드와 동일)
    frame_scores /= np.maximum(frame_counts, 1)
    frame_labels = (frame_scores >= threshold).astype(np.int32)
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
        video_name = os.path.splitext(fname)[0]
        feat_path = os.path.join(feat_root_folder, f"{video_name}.npy")

        if not os.path.isfile(feat_path):
            print(f"  ⚠️ Feature not found: {feat_path} (skip)")
            continue

        try:
            frame_labels = frame_level_inference_from_npy(
                video_path, feat_path, model, threshold=threshold
            )

            # ===== 프레임 단위 CSV 저장 =====
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
