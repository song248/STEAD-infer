import os
import torch
import numpy as np
from tqdm import tqdm
from multiprocessing import Process
from pytorchvideo.data.encoded_video import EncodedVideo
from torch import nn
import torch.nn.functional as F
import av
from contextlib import nullcontext

# ====== CONFIG ======
VIDEO_ROOT = 'my_video'          # 입력 루트: my_video/{normal, violence}
OUTPUT_ROOT = 'my_video_npy'     # 출력 루트: my_video_npy/{normal, assault}
CLASS_MAP = {                    # 입력 폴더명 -> 출력 폴더명 매핑
    'normal': 'normal',
    'violence': 'assault',
}
MODEL_NAME = 'x3d_l'
GPUS = [0, 1]

# (기존 16 → 8로 하향: 필요 시 4까지 낮추세요)  :contentReference[oaicite:2]{index=2}
BATCH_SIZE = 8

# X3D-L 전처리 파라미터 (원본 의도 유지)  :contentReference[oaicite:3]{index=3}
SIDE_SIZE = 320
CROP_SIZE = 320
NUM_FRAMES = 4
SAMPLING_RATE = 5
MEAN = torch.tensor([0.45, 0.45, 0.45]).view(1, 3, 1, 1)   # (1,C,1,1)
STD  = torch.tensor([0.225, 0.225, 0.225]).view(1, 3, 1, 1)

# AMP 사용 여부 (True 권장)
USE_AMP = True


# ====== Helpers ======
class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
    def forward(self, x):
        return x.permute(*self.dims)


def get_fps(video_path, default=30.0):
    try:
        container = av.open(video_path)
        for stream in container.streams:
            if stream.type == 'video' and stream.average_rate:
                return float(stream.average_rate)
    except:
        pass
    return default


@torch.no_grad()
def temporal_subsample(x, num_samples):
    """
    x: (T, C, H, W) or (C, T, H, W) → 반환은 (T, C, H, W)
    균일 간격으로 num_samples 개 프레임 선택
    """
    if x.dim() != 4:
        raise ValueError("expected 4D video tensor")
    # to (T, C, H, W)
    if x.shape[0] in (1, 3):   # (C, T, H, W)
        x = x.permute(1, 0, 2, 3)
    T = x.shape[0]
    if T == num_samples:
        return x
    idx = torch.linspace(0, max(T - 1, 0), steps=num_samples).round().long()
    idx = torch.clamp(idx, 0, T - 1)
    return x.index_select(0, idx)


@torch.no_grad()
def short_side_resize(x, size):
    """
    x: (T, C, H, W), 짧은 변을 size로 유지하며 종횡비 보존 리사이즈
    """
    T, C, H, W = x.shape
    if H <= W:
        new_h, new_w = size, int(round(W * size / H))
    else:
        new_h, new_w = int(round(H * size / W)), size
    # (T,C,H,W) → (T*C,1,H,W)로 펼쳐서 한 번에 interpolate
    x = x.reshape(T * C, 1, H, W)
    x = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)
    x = x.reshape(T, C, new_h, new_w)
    return x


@torch.no_grad()
def center_crop(x, crop_size):
    """
    x: (T, C, H, W) → 센터 크롭 (crop_size, crop_size)
    """
    T, C, H, W = x.shape
    if H < crop_size or W < crop_size:
        side = max(crop_size, min(H, W))
        x = short_side_resize(x, side)
        T, C, H, W = x.shape
    top = (H - crop_size) // 2
    left = (W - crop_size) // 2
    return x[:, :, top:top + crop_size, left:left + crop_size]


@torch.no_grad()
def normalize_01(x):
    # 입력: (T, C, H, W), uint8 또는 float → 0~1 float
    if x.dtype != torch.float32:
        x = x.float()
    return x / 255.0


@torch.no_grad()
def standardize(x):
    # 채널 정규화: (x - mean)/std, x: (T,C,H,W)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


@torch.no_grad()
def preprocess_clip(video_dict):
    """
    EncodedVideo.get_clip(...)의 반환 dict 이용.
    반환: (C, T, H, W) 텐서 (X3D 입력 형상). **CPU 텐서로 유지!**
    (원본은 여기서 .to(device)로 GPU에 올려 누적하던 것이 OOM 원인)  :contentReference[oaicite:4]{index=4}
    """
    if "video" not in video_dict or video_dict["video"] is None:
        return None
    v = video_dict["video"]  # pytorchvideo는 (C,T,H,W) 또는 (T,H,W,C)일 수 있음
    if v.dim() != 4:
        return None

    # to (T, C, H, W)
    if v.shape[0] in (1, 3):   # (C, T, H, W)
        v = v.permute(1, 0, 2, 3)
    elif v.shape[-1] in (1, 3):  # (T, H, W, C)
        v = v.permute(0, 3, 1, 2)
    else:
        return None

    # 1) 시간 균일 샘플링
    v = temporal_subsample(v, NUM_FRAMES)          # (T,C,H,W)
    # 2) 0~1 스케일
    v = normalize_01(v)                            # (T,C,H,W)
    # 3) 리사이즈/크롭/정규화
    v = short_side_resize(v, SIDE_SIZE)            # (T,C,h,w)
    v = center_crop(v, CROP_SIZE)                  # (T,C,CROP,CROP)
    v = standardize(v)                             # (T,C,*,*)
    # 4) (C,T,H,W)로 변환
    v = v.permute(1, 0, 2, 3).contiguous()         # (C,T,H,W)
    return v  # <-- CPU 텐서 그대로 반환 (여기서 GPU로 올리지 않음)


def init_model(gpu_id):
    torch.cuda.set_device(gpu_id)
    model = torch.hub.load('facebookresearch/pytorchvideo', MODEL_NAME, pretrained=True)
    model = model.to(f'cuda:{gpu_id}').eval()
    # 분류 헤드 제거 → 피처 추출용 (원본 동일)  :contentReference[oaicite:5]{index=5}
    del model.blocks[-1]
    return model


def list_videos_recursive(root, exts={'.mp4', '.avi', '.mov', '.mkv'}):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if os.path.splitext(fn)[1].lower() in exts:
                yield os.path.join(dirpath, fn)


def build_jobs():
    jobs = []
    for in_cls, out_cls in CLASS_MAP.items():
        in_dir = os.path.join(VIDEO_ROOT, in_cls)
        for vpath in list_videos_recursive(in_dir):
            rel = os.path.relpath(vpath, in_dir)  # 하위구조 보존
            spath = os.path.join(OUTPUT_ROOT, out_cls, os.path.splitext(rel)[0] + '.npy')
            jobs.append((vpath, spath))
    return jobs


def process_video_batch(video_paths, gpu_id):
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(gpu_id)
    model = init_model(gpu_id)
    amp_ctx = torch.cuda.amp.autocast if USE_AMP else nullcontext

    print(f"[GPU {gpu_id}] Start {len(video_paths)} videos")

    for video_path, save_path in video_paths:
        try:
            fps = get_fps(video_path)
            clip_duration = (NUM_FRAMES * SAMPLING_RATE) / max(fps, 1e-6)

            video = EncodedVideo.from_path(video_path)
            if video.duration is None or video.duration < clip_duration:
                print(f"[GPU {gpu_id}] [SKIP] {video_path}: too short")
                continue

            total_clips = int(video.duration // clip_duration)

            feats = []
            # === 스트리밍 배치 처리: 클립을 쌓아두지 않고 즉시 배치 구성 ===
            for i in range(0, total_clips, BATCH_SIZE):
                batch_cpu = []
                for j in range(i, min(i + BATCH_SIZE, total_clips)):
                    clip = video.get_clip(
                        start_sec=j * clip_duration,
                        end_sec=(j + 1) * clip_duration
                    )
                    v = preprocess_clip(clip)  # CPU 텐서
                    if v is not None:
                        batch_cpu.append(v)

                if not batch_cpu:
                    continue

                # CPU에서 스택 → pinned memory → GPU 전송
                batch = torch.stack(batch_cpu, dim=0)  # CPU (B,C,T,H,W)
                try:
                    batch = batch.pin_memory()
                except Exception:
                    pass
                batch = batch.to(device, non_blocking=True)

                with torch.inference_mode(), amp_ctx():
                    pred = model(batch)   # 형태: (B, ...)
                feats.append(pred.detach().cpu().numpy())

                # 즉시 정리
                del batch, batch_cpu, pred
                torch.cuda.empty_cache()

            if not feats:
                print(f"[GPU {gpu_id}] [SKIP] {video_path}: no valid clips")
                # 비디오 객체 정리
                try:
                    del video
                except Exception:
                    pass
                torch.cuda.empty_cache()
                continue

            feats = np.concatenate(feats, axis=0)  # (N_clips, ...)
            reduced = np.max(feats, axis=0)        # 클립 축 max-pooling

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, reduced)
            print(f"[GPU {gpu_id}] ✅ Saved: {save_path}")

            # 파일 단위 캐시 정리
            del video, feats, reduced
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"[GPU {gpu_id}] [ERROR] {video_path}: {e}")
            torch.cuda.empty_cache()


def main():
    jobs = build_jobs()
    print(f"🚀 Total {len(jobs)} videos across {len(GPUS)} GPUs")
    # 라운드로빈 분배 (원 코드와 동일 아이디어)  :contentReference[oaicite:6]{index=6}
    splits = [[] for _ in range(len(GPUS))]
    for i, job in enumerate(jobs):
        splits[i % len(GPUS)].append(job)

    procs = []
    for gpu_id, sub in zip(GPUS, splits):
        p = Process(target=process_video_batch, args=(sub, gpu_id))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
