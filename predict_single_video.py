import os
import torch
import numpy as np
from model import Model
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose, Lambda, CenterCrop, Normalize
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)
from torch import nn
from tqdm import tqdm
import av

VIDEO_PATH = 'assault005.mp4'
MODEL_PATH = 'saved_models/888tiny.pkl'
THRESHOLD = 0.5

DEVICES = ['cuda:0', 'cuda:1']
USE_PARALLEL = len(DEVICES) > 1

params = {
    "side_size": 320,
    "crop_size": 320,
    "num_frames": 16,
    "sampling_rate": 5,
}

def get_fps(video_path):
    try:
        container = av.open(video_path)
        for stream in container.streams:
            if stream.type == 'video':
                return float(stream.average_rate)
    except:
        pass
    return 30.0

class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)

# X3D-L transform
transform = ApplyTransformToKey(
    key="video",
    transform=Compose([
        UniformTemporalSubsample(params["num_frames"]),
        Lambda(lambda x: x / 255.0),
        Permute((1, 0, 2, 3)),  # TCHW → CTHW
        Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
        ShortSideScale(params["side_size"]),
        CenterCrop((params["crop_size"], params["crop_size"])),
        Permute((1, 0, 2, 3)),  # CTHW → TCHW
    ])
)

def extract_x3d_features(video_path, x3d_model, device):
    fps = get_fps(video_path)
    clip_duration = (params["num_frames"] * params["sampling_rate"]) / fps
    video = EncodedVideo.from_path(video_path)

    total_clips = int(float(video.duration) // clip_duration)
    print(f"\n📼 Extracting clips from {video_path} | Duration: {float(video.duration):.2f}s, Estimated clips: {total_clips}")

    features = []
    for clip_index in tqdm(range(total_clips), desc="🔄 Extracting Clips", ncols=80):
        clip = video.get_clip(
            start_sec=clip_index * clip_duration,
            end_sec=(clip_index + 1) * clip_duration
        )
        if clip is None or clip.get("video") is None:
            continue

        try:
            clip_input = transform(clip)["video"]  # already [C, T, H, W]
            input_tensor = clip_input.unsqueeze(0).to(device) 
            with torch.no_grad():
                feat = x3d_model(input_tensor)  # [1, 192, 16, 10, 10]
            features.append(feat.squeeze(0).cpu())
        except Exception as e:
            print(f"[WARNING] Clip {clip_index} failed: {e}")

    if not features:
        raise ValueError("❌ No valid clips extracted from video.")

    return torch.stack(features)  # [N, 192, 16, 10, 10]

def predict_parallel(video_path):
    # Load X3D-L backbone
    x3d_model = torch.hub.load('facebookresearch/pytorchvideo', 'x3d_l', pretrained=True)
    del x3d_model.blocks[-1]
    x3d_model.eval().to(DEVICES[0])

    # Extract X3D features
    clip_features = extract_x3d_features(video_path, x3d_model, DEVICES[0])  # [N, 192, 16, 10, 10]

    # Max-pool over time dimension (clips)
    reduced_feat = torch.max(clip_features, dim=0, keepdim=True)[0].to(DEVICES[0])  # [1, 192, 16, 10, 10]

    # Load anomaly detection model
    model = Model()
    model.load_state_dict(torch.load(MODEL_PATH))
    model = model.to(DEVICES[0])

    if USE_PARALLEL:
        model = nn.DataParallel(model, device_ids=[0, 1])

    model.eval()
    with torch.no_grad():
        score, _ = model(reduced_feat)  # [1, 1]
        prob = torch.sigmoid(score).item()

    print(f"\n🧪 Video: {video_path}")
    print(f"Abnormal score: {prob:.4f}")
    print("Result:", "🚨 Abnormal" if prob >= THRESHOLD else "✅ Normal")

if __name__ == '__main__':
    predict_parallel(VIDEO_PATH)
