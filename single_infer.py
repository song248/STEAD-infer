import torch
import numpy as np
from model import Model
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose, Lambda
from torch import nn

# ===== CONFIG =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "saved_models/888tiny.pkl"
video_file = "test01.mp4"
threshold = 0.5

# ===== X3D 설정 =====
model_name = 'x3d_l'
transform_params = {
    "num_frames": 16,
    "sampling_rate": 5,
    "side_size": 320,
    "crop_size": 320,
    "frames_per_second": 30,
}
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
clip_duration = (transform_params["num_frames"] * transform_params["sampling_rate"]) / transform_params["frames_per_second"]

# ===== Temporal Subsample =====
def temporal_subsample(frames: torch.Tensor, num_samples: int):
    t = frames.shape[0]
    if t < num_samples:
        pad = frames[-1:].repeat(num_samples - t, 1, 1, 1)
        frames = torch.cat([frames, pad], dim=0)
        t = frames.shape[0]
    indices = torch.linspace(0, t - 1, num_samples).long()
    return frames[indices]

# ===== Normalize 함수 =====
def normalize_video_tensor(video_tensor, mean, std):
    # (T, C, H, W)
    mean = torch.tensor(mean, device=video_tensor.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=video_tensor.device).view(1, -1, 1, 1)
    return (video_tensor - mean) / std

# ===== 전처리 Transform =====
transform = Compose([
    Lambda(lambda x: temporal_subsample(x, transform_params["num_frames"])),     # (T, H, W, C)
    Lambda(lambda x: x.permute(0, 3, 1, 2) / 255.0),                               # (T, C, H, W)
    Lambda(lambda x: normalize_video_tensor(x, mean, std)),                       # Normalize
    Lambda(lambda x: torch.nn.functional.interpolate(
        x, size=(transform_params["side_size"], transform_params["side_size"]),
        mode="bilinear", align_corners=False)),                                   # Resize
    Lambda(lambda x: x[:, :,                                                         
                       transform_params["side_size"]//2 - transform_params["crop_size"]//2:
                       transform_params["side_size"]//2 + transform_params["crop_size"]//2,
                       transform_params["side_size"]//2 - transform_params["crop_size"]//2:
                       transform_params["side_size"]//2 + transform_params["crop_size"]//2]),  # Center crop
    Lambda(lambda x: x.permute(1, 0, 2, 3)),  # (T, C, H, W) → (C, T, H, W)
])

# ===== X3D 모델 로딩 =====
x3d = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)
x3d = nn.Sequential(*list(x3d.blocks[:5]))  # x3d.blocks[5:]은 classification head
x3d = x3d.to(device).eval()

# ===== FEATURE 추출 =====
def extract_x3d_feature(path):
    video = EncodedVideo.from_path(path)
    clip = video.get_clip(start_sec=0, end_sec=clip_duration)
    frames = clip["video"]  # shape may vary

    # shape 정리: (C, T, H, W) → (T, H, W, C)
    if frames.shape[0] == 3:
        frames = frames.permute(1, 2, 3, 0)
    elif frames.shape[-1] != 3:
        raise ValueError(f"Unsupported frame shape: {frames.shape}")

    frames = transform(frames).unsqueeze(0).to(device)  # (1, C, T, H, W)
    with torch.no_grad():
        feat = x3d(frames).squeeze(0)  # (C, T, H, W)
    return feat

# ===== STEAD 이상 판단 =====
def predict_anomaly(feature, model_path):
    model = Model(ff_mult=1, dims=(32, 32), depths=(1, 1)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    with torch.no_grad():
        score, _ = model(feature.unsqueeze(0).to(device))  # (1, C, T, H, W)
        prob = torch.sigmoid(score).item()
    return prob

# ===== MAIN =====
if __name__ == "__main__":
    print(f"🔍 Extracting feature from: {video_file}")
    feature = extract_x3d_feature(video_file)

    print("🧠 Running anomaly prediction...")
    score = predict_anomaly(feature, model_path)
    result = "Abnormal" if score > threshold else "Normal"

    print(f"\n🎯 Anomaly Score: {score:.4f}")
    print(f"👉 Result: {result}")
