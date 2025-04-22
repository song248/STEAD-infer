import os
import torch
import numpy as np
from tqdm import tqdm
from multiprocessing import Process
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose, Lambda, CenterCrop, Normalize
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample
)
from torch import nn
import av

# ====== CONFIG ======
VIDEO_DIR = 'violence'
OUTPUT_DIR = 'violence_npy'
MODEL_NAME = 'x3d_l'
GPUS = [0, 1]
BATCH_SIZE = 16  # 💡 클립을 나눠서 추론하도록 설정

model_transform_params = {
    "x3d_l": {
        "side_size": 320,
        "crop_size": 320,
        "num_frames": 4,
        "sampling_rate": 5,
    }
}
params = model_transform_params[MODEL_NAME]

# ====== TRANSFORM ======
class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
    def forward(self, x):
        return x.permute(*self.dims)

transform = ApplyTransformToKey(
    key="video",
    transform=Compose([
        UniformTemporalSubsample(params["num_frames"]),
        Lambda(lambda x: x / 255.0),
        Permute((1, 0, 2, 3)),
        Normalize([0.45] * 3, [0.225] * 3),
        ShortSideScale(params["side_size"]),
        CenterCrop((params["crop_size"], params["crop_size"])),
        Permute((1, 0, 2, 3)),
    ])
)

def get_fps(video_path):
    try:
        container = av.open(video_path)
        for stream in container.streams:
            if stream.type == 'video':
                return float(stream.average_rate)
    except:
        pass
    return 30.0

def init_model(gpu_id):
    torch.cuda.set_device(gpu_id)
    model = torch.hub.load('facebookresearch/pytorchvideo', MODEL_NAME, pretrained=True)
    model = model.to(f'cuda:{gpu_id}').eval()
    del model.blocks[-1]
    return model

def process_video_batch(video_paths, gpu_id):
    torch.cuda.set_device(gpu_id)
    model = init_model(gpu_id)
    print(f"[GPU {gpu_id}] Starting processing {len(video_paths)} videos...")

    for video_path, save_path in video_paths:
        print(f"[GPU {gpu_id}] ▶ {os.path.basename(video_path)}")
        try:
            fps = get_fps(video_path)
            clip_duration = (params["num_frames"] * params["sampling_rate"]) / fps
            video = EncodedVideo.from_path(video_path)

            if video.duration < clip_duration:
                print(f"[GPU {gpu_id}] [SKIP] {video_path}: too short")
                continue

            # 클립 transform 먼저 수행
            clip_inputs = []
            total_clips = int(video.duration // clip_duration)
            for idx in range(total_clips):
                clip = video.get_clip(
                    start_sec=idx * clip_duration,
                    end_sec=(idx + 1) * clip_duration
                )
                if clip is None or clip.get("video") is None:
                    break
                try:
                    transformed = transform(clip)["video"]
                    clip_inputs.append(transformed)
                except Exception as e:
                    print(f"[GPU {gpu_id}] [Clip {idx}] transform failed: {e}")

            if not clip_inputs:
                print(f"[GPU {gpu_id}] [SKIP] {video_path}: no valid clips")
                continue

            # 클립들을 BATCH_SIZE 단위로 나눠서 추론
            print(f"[GPU {gpu_id}] 🔁 Inference on {len(clip_inputs)} clips (batched)...")
            features = []
            for i in range(0, len(clip_inputs), BATCH_SIZE):
                batch = torch.stack(clip_inputs[i:i+BATCH_SIZE]).to(f'cuda:{gpu_id}')
                with torch.no_grad():
                    preds = model(batch)  # shape: (B, C, T, H, W)
                features.extend(preds.cpu().numpy())

            features = np.stack(features)
            reduced_feature = np.max(features, axis=0)

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, reduced_feature)

            print(f"[GPU {gpu_id}] ✅ Saved: {save_path}")

        except Exception as e:
            print(f"[GPU {gpu_id}] [ERROR] {video_path}: {e}")

def get_video_list():
    videos = []
    for f in os.listdir(VIDEO_DIR):
        if f.endswith('.mp4'):
            video_path = os.path.join(VIDEO_DIR, f)
            save_path = os.path.join(OUTPUT_DIR, f.replace('.mp4', '.npy'))
            videos.append((video_path, save_path))
    return videos

def main():
    all_jobs = get_video_list()
    print(f"🚀 Total {len(all_jobs)} videos to process across {len(GPUS)} GPUs")

    # 작업 분배
    split_jobs = [[] for _ in range(len(GPUS))]
    for idx, job in enumerate(all_jobs):
        split_jobs[idx % len(GPUS)].append(job)

    processes = []
    for gpu_id, job_list in zip(GPUS, split_jobs):
        p = Process(target=process_video_batch, args=(job_list, gpu_id))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
