# main.py — safe fine-tuning save (no overwrite of pretrained), lr cast to float
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from torch import nn
import numpy as np
from utils import save_best_record
from timm.scheduler.cosine_lr import CosineLRScheduler
from torchinfo import summary
from tqdm import tqdm
import option
args = option.parse_args()
from model import Model
from dataset import Dataset
from train import train
from test import test
import datetime
import os
import random


def save_config(save_path):
    path = save_path + '/'
    os.makedirs(path, exist_ok=True)
    f = open(path + "config_{}.txt".format(datetime.datetime.now()), 'w')
    for key in vars(args).keys():
        f.write('{}: {}'.format(key, vars(args)[key]))
        f.write('\n')


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv1d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.Conv3d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


def make_save_dir_and_prefix(args):
    """
    기존 코드의 저장 폴더 규칙을 유지하되, 파일명에는 항상 타임스탬프와
    프리트레인 파일명을 접두사에 포함시켜 덮어쓰기 위험을 제거합니다.
    """
    # 기존 규칙 유지: ./ckpt/{lr}_{batch}_{comment}
    lr_str = str(args.lr)
    save_dir = './ckpt/{}'.format(args.comment)
    os.makedirs(save_dir, exist_ok=True)

    # 프리트레인 접두사
    pt_prefix = None
    if getattr(args, 'pretrained_ckpt', None):
        base = os.path.basename(args.pretrained_ckpt)
        pt_prefix = os.path.splitext(base)[0]  # e.g., 888tiny

    # 접두사: {model_name}_ft-from-{pt}_YYYYMMDD-HHMMSS
    t = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    if pt_prefix:
        prefix = f"{args.model_name}_ft-from-{pt_prefix}_{t}"
    else:
        prefix = f"{args.model_name}_{t}"
    return save_dir, prefix


if __name__ == '__main__':
    args = option.parse_args()
    random.seed(2025)
    np.random.seed(2025)
    torch.cuda.manual_seed(2025)
    device = torch.device('cuda')

    # 저장 디렉터리/접두사 생성 및 설정 기록
    save_dir, prefix = make_save_dir_and_prefix(args)
    save_config(save_dir)

    # DO NOT SHUFFLE, shuffling is handled by the Dataset class and not the DataLoader
    train_loader = DataLoader(Dataset(args, test_mode=False),
                              batch_size=args.batch_size // 2)
    test_loader = DataLoader(Dataset(args, test_mode=True),
                             batch_size=args.batch_size)

    model = Model(dropout=args.dropout_rate, attn_dropout=args.attn_dropout_rate)

    # 프리트레인 로드(읽기 전용), 없으면 가중치 초기화
    if args.pretrained_ckpt:
        ckpt = torch.load(args.pretrained_ckpt, map_location='cpu')
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        print("pretrained loaded from", args.pretrained_ckpt)
        if missing:
            print("[load_state_dict] missing keys:", missing)
        if unexpected:
            print("[load_state_dict] unexpected keys:", unexpected)
    else:
        model.apply(init_weights)

    model = model.to(device)

    # lr 문자열 안전 처리
    try:
        lr_value = float(args.lr)
    except Exception:
        # 혹시라도 변환 실패 시 기본값 사용
        lr_value = 2e-4
        print(f"[warn] failed to parse lr='{args.lr}', fallback to {lr_value}")

    optimizer = optim.AdamW(model.parameters(), lr=lr_value, weight_decay=0.2)

    num_steps = len(train_loader)
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=args.max_epoch * num_steps,
        cycle_mul=1.,
        lr_min=lr_value * 0.2,
        warmup_lr_init=lr_value * 0.01,
        warmup_t=args.warmup * num_steps,
        cycle_limit=20,
        t_in_epochs=False,
        warmup_prefix=True,
        cycle_decay=0.95,
    )

    test_info = {"epoch": [], "test_AUC": [], "test_PR": []}

    for step in tqdm(
        range(0, args.max_epoch),
        total=args.max_epoch,
        dynamic_ncols=True
    ):
        cost = train(train_loader, model, optimizer, scheduler, device, step)
        scheduler.step(step + 1)

        auc, pr_auc = test(test_loader, model, args, device)

        test_info["epoch"].append(step)
        test_info["test_AUC"].append(auc)
        test_info["test_PR"].append(pr_auc)

        # 에폭별 저장: {save_dir}/{prefix}_ep{step}.pkl
        epoch_path = os.path.join(save_dir, f"{prefix}_ep{step}.pkl")
        torch.save(model.state_dict(), epoch_path)

        # 기록 저장
        rec_path = os.path.join(save_dir, f'{prefix}_ep{step}.txt')
        save_best_record(test_info, rec_path)

    # 최종 저장: {save_dir}/{prefix}_final.pkl  (프리트레인과 경로가 같을 가능성은 사실상 없음)
    final_path = os.path.join(save_dir, f"{prefix}_final.pkl")
    torch.save(model.state_dict(), final_path)
    print("saved final to:", final_path)
