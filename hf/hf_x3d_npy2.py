import os
import math
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from pytorchvideo.data.encoded_video import EncodedVideo

# ===== CONFIG =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

video_root_folder = "hf-violence/video"
feature_save_folder = "hf_npy"  # 특징 저장 폴더
os.makedirs(feature_save_folder, exist_ok=True)

valid_exts = [".mp4", ".avi"]

# ===== X3D 설정 =====
model_name = 'x3d_l'
transform_params = {
    # 1초 창, 10fps 입력 (30fps 원본에서 3프레임마다 1프레임 샘플)
    "num_frames": 10,        # ← 15 -> 10
    "sampling_rate": 3,      # ← 4  -> 3   (10*3/30 = 1.0초)
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

# ===== 클립별 특징 추출 및 저장 =====
def extract_and_save_features(video_path, save_path):
    video = EncodedVideo.from_path(video_path)
    fps = transform_params["frames_per_second"]
    duration = float(video.duration)

    # 1초 창: 10프레임 × 3샘플링 / 30fps = 1.0s
    window_sec = (transform_params["num_frames"] * transform_params["sampling_rate"]) / fps  # = 1.0
    # 0.5초 간격 슬라이딩
    stride_sec = 0.5

    features = []

    total_steps = max(1, math.ceil(max(0.0, (duration - window_sec)) / stride_sec) + 1)
    for step in tqdm(range(total_steps), desc="Extracting features", leave=False, ncols=80):
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

        # 30fps 기준 3프레임 간격으로 10프레임 뽑아 10fps 입력 생성
        frames = fixed_interval_sample(
            frames, transform_params["sampling_rate"], transform_params["num_frames"])
        frames = preprocess_frames(frames).unsqueeze(0).to(device)

        with torch.no_grad():
            feature = x3d(frames).squeeze(0).cpu().numpy()  # (C,T,H,W) → np.array
            features.append(feature)

    features = np.stack(features)  # (N_clips, C, T, H, W)
    np.save(save_path, features)
    print(f"✅ Saved: {save_path}")

# ===== MAIN =====
if __name__ == "__main__":
    video_files = [
        fname for fname in os.listdir(video_root_folder)
        if os.path.splitext(fname)[-1].lower() in valid_exts
    ]

    for idx, fname in enumerate(video_files, start=1):
        print(f"\n[{idx}/{len(video_files)}] Processing {fname}")
        video_path = os.path.join(video_root_folder, fname)
        video_name = os.path.splitext(fname)[0]
        save_path = os.path.join(feature_save_folder, f"{video_name}.npy")

        try:
            extract_and_save_features(video_path, save_path)
        except Exception as e:
            print(f"🚨 Error: {e}")
            continue

    print(f"\n✅ All features saved in folder: {feature_save_folder}")
