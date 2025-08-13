#!/usr/bin/env python3
# makelist.py — build train/test txt from my own npy features

import os
import argparse
import random
from pathlib import Path

def gather_npy(dirpath: Path):
    return sorted([str(p) for p in dirpath.rglob("*.npy")])

def write_list(path, abn_list, nor_list):
    """
    Keep abnormal first, then normal (Dataset이 이 순서를 기대하는 구조를 쓰는 경우가 많음)
    """
    with open(path, "w") as f:
        for p in abn_list:
            f.write(p + "\n")
        for p in nor_list:
            f.write(p + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="my_video_npy",
                    help="npy 루트 폴더 (예: my_video_npy)")
    ap.add_argument("--abnormal", type=str, default="assault",
                    help="비정상 클래스 폴더 이름")
    ap.add_argument("--normal", type=str, default="normal",
                    help="정상 클래스 폴더 이름")
    ap.add_argument("--train_out", type=str, default="train_list.txt",
                    help="학습 리스트 출력 파일")
    ap.add_argument("--test_out", type=str, default="test_list.txt",
                    help="테스트 리스트 출력 파일")
    ap.add_argument("--split", type=float, default=0.8,
                    help="train 비율 (0~1)")
    ap.add_argument("--seed", type=int, default=2025,
                    help="셔플 시드")
    ap.add_argument("--absolute", action="store_true",
                    help="경로를 절대경로로 기록 (기본은 상대경로)")
    args = ap.parse_args()

    root = Path(args.root)
    abn_dir = root / args.abnormal
    nor_dir = root / args.normal

    # 존재 확인
    if not abn_dir.exists():
        raise FileNotFoundError(f"Not found: {abn_dir}")
    if not nor_dir.exists():
        raise FileNotFoundError(f"Not found: {nor_dir}")

    # 파일 수집
    abn = gather_npy(abn_dir)
    nor = gather_npy(nor_dir)

    if args.absolute:
        abn = [str(Path(p).resolve()) for p in abn]
        nor = [str(Path(p).resolve()) for p in nor]
    else:
        # 상대 경로로 통일
        abn = [str(Path(p)) for p in abn]
        nor = [str(Path(p)) for p in nor]

    # 셔플 & 스플릿
    random.seed(args.seed)
    random.shuffle(abn)
    random.shuffle(nor)

    def split(xs):
        k = max(1, int(len(xs) * args.split)) if xs else 0
        return xs[:k], xs[k:]

    abn_tr, abn_te = split(abn)
    nor_tr, nor_te = split(nor)

    # 출력 (항상 "비정상 먼저 → 정상" 순서)
    write_list(args.train_out, abn_tr, nor_tr)
    write_list(args.test_out, abn_te, nor_te)

    print(f"[DONE] train: {len(abn_tr)+len(nor_tr)} "
          f"(abn {len(abn_tr)}, nor {len(nor_tr)})")
    print(f"[DONE] test : {len(abn_te)+len(nor_te)} "
          f"(abn {len(abn_te)}, nor {len(nor_te)})")
    print(f"Files written -> {args.train_out}, {args.test_out}")

if __name__ == "__main__":
    main()
