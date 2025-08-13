import os
import math
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from pytorchvideo.data.encoded_video import EncodedVideo

# =======================
# GPU 고정: 1번 GPU만 사용
# =======================
# 주의: 이 설정은 torch의 CUDA 컨텍스트가 생성되기 전에 설정되어야 합니다.
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# ===== CONFIG =====
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 입력/출력 폴더 설정
input_root_folder = "my_video_ver2"          # my_video/{normal, violence}
output_root_folder = "my_video_ver2_npy"     # my_video_npy/{normal, assault}

# 클래스 매핑: 입력 폴더명 -> 출력 폴더명
class_map = {
    "normal": "normal",
    "violence": "assault",
}

os.makedirs(output_root_folder, exist_ok=True)
for _in, _out in class_map.items():
    os.makedirs(os.path.join(output_root_folder, _out), exist_ok=True)

valid_exts = [".mp4", ".avi", ".mov", ".mkv"]

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
    # 입력 frames: (T,H,W,C) 또는 (3,T,H,W)에서 변환됨
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
# 분류 헤드를 제거하여 중간 특징만 추출
x3d = torch.hub.load('facebookresearch/pytorchvideo', model_name, pretrained=True)
x3d = nn.Sequential(*list(x3d.blocks[:5]))  # classification head 제거
x3d = x3d.to(device).eval()

# ===== 클립별 특징 추출 및 저장 =====
def extract_and_save_features(video_path, save_path):
    video = EncodedVideo.from_path(video_path)
    fps = transform_params["frames_per_second"]
    duration = float(video.duration)
    window_sec = (transform_params["num_frames"] * transform_params["sampling_rate"]) / fps
    stride_sec = 1.0

    features = []

    total_steps = max(1, math.ceil((duration - window_sec) / stride_sec) + 1)
    for step in tqdm(range(total_steps), desc="Extracting features", leave=False, ncols=80):
        start = step * stride_sec
        end = start + window_sec

        try:
            clip = video.get_clip(start_sec=start, end_sec=end)
        except Exception:
            break

        frames = clip["video"]
        # pytorchvideo는 (C,T,H,W) 형식일 수 있음
        if frames.shape[0] == 3:
            frames = frames.permute(1, 2, 3, 0)  # (C,T,H,W)→(T,H,W,C)
        elif frames.shape[-1] != 3:
            raise ValueError(f"Unsupported frame shape: {frames.shape}")

        frames = fixed_interval_sample(frames, transform_params["sampling_rate"], transform_params["num_frames"])
        frames = preprocess_frames(frames).unsqueeze(0).to(device)

        with torch.no_grad():
            feature = x3d(frames).squeeze(0).cpu().numpy()  # (C,T,H,W) → np.array
            features.append(feature)

    if len(features) == 0:
        print(f"⚠️  No features extracted for: {video_path} (skipped)")
        return False

    features = np.stack(features)  # (N_clips, C, T, H, W)
    np.save(save_path, features)
    print(f"✅ Saved: {save_path}")
    return True

# ===== MAIN =====
if __name__ == "__main__":
    total_videos = 0
    processed = 0

    for in_cls, out_cls in class_map.items():
        in_dir = os.path.join(input_root_folder, in_cls)
        out_dir = os.path.join(output_root_folder, out_cls)
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.isdir(in_dir):
            print(f"⚠️  Input class folder not found: {in_dir} (skipping)")
            continue

        video_files = [
            f for f in sorted(os.listdir(in_dir))
            if os.path.splitext(f)[-1].lower() in valid_exts
        ]

        print(f"\n=== Class: {in_cls} → {out_cls} | {len(video_files)} files ===")
        total_videos += len(video_files)

        for idx, fname in enumerate(video_files, start=1):
            print(f"[{idx}/{len(video_files)}] Processing {fname}")
            video_path = os.path.join(in_dir, fname)
            video_name = os.path.splitext(fname)[0]
            save_path = os.path.join(out_dir, f"{video_name}.npy")

            try:
                ok = extract_and_save_features(video_path, save_path)
                if ok:
                    processed += 1
            except Exception as e:
                print(f"🚨 Error on {fname}: {e}")
                continue

    print(f"\n✅ Done. {processed}/{total_videos} videos processed.")
