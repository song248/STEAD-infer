import torch
import math
from torch import nn
import numpy as np
from model import Model
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose, Lambda

# ===== CONFIG =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "saved_models/888tiny.pkl"
video_file = "test01.mp4"
threshold = 0.5

# ===== X3D 설정 =====
model_name = 'x3d_l'
transform_params = {
    "num_frames": 15,           # 15프레임
    "sampling_rate": 4,         # 4프레임 간격
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
    frames = frames.permute(0, 3, 1, 2) / 255.0  # (T, H, W, C) → (T, C, H, W)
    frames = normalize_video_tensor(frames, mean, std)
    frames = torch.nn.functional.interpolate(
        frames, size=(transform_params["side_size"], transform_params["side_size"]),
        mode="bilinear", align_corners=False
    )
    crop = transform_params["crop_size"]
    center = transform_params["side_size"] // 2
    frames = frames[:, :, center - crop//2:center + crop//2,
                          center - crop//2:center + crop//2]
    frames = frames.permute(1, 0, 2, 3)  # (T, C, H, W) → (C, T, H, W)
    return frames

# ===== X3D 모델 로딩 =====
x3d = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)
x3d = nn.Sequential(*list(x3d.blocks[:5]))  # classification head 제거
x3d = x3d.to(device).eval()

# ===== 이상 판단 함수 =====
def predict_anomaly(feature, model_path):
    model = Model(ff_mult=1, dims=(32, 32), depths=(1, 1)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    with torch.no_grad():
        score, _ = model(feature.unsqueeze(0).to(device))  # (1, C, T, H, W)
        prob = torch.sigmoid(score).item()
    return prob

# ===== 실시간 슬라이딩 인퍼런스 =====
def stream_anomaly_inference(path):
    video = EncodedVideo.from_path(path)
    fps = transform_params["frames_per_second"]
    duration = float(video.duration)
    window_sec = (transform_params["num_frames"] * transform_params["sampling_rate"]) / fps  # 2초
    stride_sec = 1.0  # 슬라이딩 간격

    total_steps = math.ceil((duration - window_sec) / stride_sec) + 1
    print(f"📽 Video duration: {duration:.2f}s")
    print(f"🔄 Sliding window: {window_sec:.2f}s window every {stride_sec:.1f}s → {total_steps} steps\n")

    for step in range(total_steps):
        start = step * stride_sec
        end = start + window_sec
        try:
            clip = video.get_clip(start_sec=start, end_sec=end)
        except:
            break
        frames = clip["video"]  # (C, T, H, W) or (T, H, W, C)
        if frames.shape[0] == 3:
            frames = frames.permute(1, 2, 3, 0)  # → (T, H, W, C)
        elif frames.shape[-1] != 3:
            raise ValueError(f"Unsupported frame shape: {frames.shape}")

        frames = fixed_interval_sample(frames, transform_params["sampling_rate"], transform_params["num_frames"])
        frames = preprocess_frames(frames).unsqueeze(0).to(device)

        with torch.no_grad():
            feature = x3d(frames).squeeze(0)
            score = predict_anomaly(feature, model_path)
            label = "Abnormal" if score > threshold else "Normal"
            print(f"[{start:.1f}s ~ {end:.1f}s]  Score: {score:.4f}  → {label}")

# ===== MAIN =====
if __name__ == "__main__":
    print(f"🚀 Running sliding window inference on video: {video_file}")
    stream_anomaly_inference(video_file)
