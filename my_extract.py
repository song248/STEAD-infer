import os
import torch
import numpy as np
from tqdm import tqdm
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose, Lambda, CenterCrop, Normalize
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)
from torch import nn
import av

# ====== Config ======
VIDEO_DIR = 'violence'
OUTPUT_DIR = 'my_video_npy'
AVAILABLE_GPUS = [0, 1]  # 사용할 GPU 인덱스
MODEL_NAME = 'x3d_l'

# ====== Load pretrained model and remove classification head ======
model = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)
model = model.to(device).eval()
del model.blocks[-1]  # remove classification head

# ====== Model-specific transform parameters ======
model_transform_params = {
    "x3d_l": {
        "side_size": 320,
        "crop_size": 320,
        "num_frames": 16,
        "sampling_rate": 5,
    }
}
params = model_transform_params[model_name]

# ====== Permute helper ======
class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)

# ====== Transform 정의 ======
transform = ApplyTransformToKey(
    key="video",
    transform=Compose([
        UniformTemporalSubsample(params["num_frames"]),
        Lambda(lambda x: x / 255.0),
        Permute((1, 0, 2, 3)),  # (T, H, W, C) → (C, T, H, W)
        Normalize([0.45, 0.45, 0.45], [0.225, 0.225, 0.225]),
        ShortSideScale(size=params["side_size"]),
        CenterCrop((params["crop_size"], params["crop_size"])),
        Permute((1, 0, 2, 3)),  # (C, T, H, W) → (T, C, H, W)
    ])
)

# ====== FPS 추출 ======
def get_fps(video_path):
    try:
        container = av.open(video_path)
        for stream in container.streams:
            if stream.type == 'video':
                return float(stream.average_rate)
    except:
        pass
    return 30.0  # fallback

# ====== Process a single video ======
def process_video(video_path, save_path):
    try:
        fps = get_fps(video_path)
        clip_duration = (params["num_frames"] * params["sampling_rate"]) / fps
        video = EncodedVideo.from_path(video_path)

        if video.duration < clip_duration:
            print(f"[SKIPPED] {video_path}: too short ({video.duration:.2f}s)")
            return

        features = []
        clip_index = 0

        while True:
            clip = video.get_clip(
                start_sec=clip_index * clip_duration,
                end_sec=(clip_index + 1) * clip_duration
            )
            if clip is None or clip.get("video") is None:
                break

            try:
                clip_input = transform(clip)
                with torch.no_grad():
                    pred = model(clip_input["video"].unsqueeze(0).to(device))  # shape: [1, C, T, H, W]
                features.append(pred.squeeze(0).cpu().numpy())
            except Exception as inner_e:
                print(f"[WARNING] Clip {clip_index} failed in {video_path}: {inner_e}")

            clip_index += 1

        if not features:
            print(f"[SKIPPED] {video_path}: no valid clips")
            return

        # 👉 Abuse001_x264.npy와 동일한 포맷: max-pooling across clips
        features = np.stack(features)  # shape: (N_clips, C, T, H, W)
        reduced_feature = np.max(features, axis=0)  # shape: (C, T, H, W)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, reduced_feature)

    except Exception as e:
        print(f"[ERROR] {video_path}: {e}")

# ====== Main Loop ======
def main():
    for class_name in os.listdir(VIDEO_DIR):
        class_dir = os.path.join(VIDEO_DIR, class_name)
        if not os.path.isdir(class_dir):
            continue
        for file_name in tqdm(os.listdir(class_dir), desc=class_name):
            if not file_name.endswith(".mp4"):
                continue
            video_path = os.path.join(class_dir, file_name)
            save_path = os.path.join(OUTPUT_DIR, class_name, file_name.replace('.mp4', '.npy'))
            process_video(video_path, save_path)

if __name__ == "__main__":
    main()
