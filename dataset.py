import torch.utils.data as data
import numpy as np
import torch
import random
torch.set_float32_matmul_precision('medium')
import option
args = option.parse_args()

def is_normal(path: str):
    p = path.lower()
    return "normal" in p  # 대소문자 무시, 내 데이터에 맞춤

class Dataset(data.Dataset):
    def __init__(self, args, test_mode=False):
        self.test_mode = test_mode
        self.rgb_list_file = args.test_rgb_list if test_mode else args.rgb_list
        self.list = [l.strip() for l in open(self.rgb_list_file) if l.strip()]

        # 리스트에서 정상/비정상 인덱스 동적으로 수집 (하드코드 제거)
        self.n_all = [i for i, l in enumerate(self.list) if is_normal(l)]
        self.a_all = [i for i, l in enumerate(self.list) if not is_normal(l)]

        if not test_mode:
            # 첫 에폭용 셔플 큐 준비
            self.n_ind = self.n_all.copy()
            self.a_ind = self.a_all.copy()
            random.shuffle(self.n_ind)
            random.shuffle(self.a_ind)

    def __getitem__(self, index):
        if not self.test_mode:
            # 고갈되면 새로 채워 셔플
            if len(self.n_ind) == 0 or len(self.a_ind) == 0:
                self.n_ind = self.n_all.copy()
                self.a_ind = self.a_all.copy()
                random.shuffle(self.n_ind)
                random.shuffle(self.a_ind)

            nindex = self.n_ind.pop()
            aindex = self.a_ind.pop()

            npath = self.list[nindex]
            apath = self.list[aindex]

            nfeatures = np.array(np.load(npath, allow_pickle=True), dtype=np.float32)
            afeatures = np.array(np.load(apath, allow_pickle=True), dtype=np.float32)

            nlabel = 0.0 if is_normal(npath) else 1.0
            alabel = 0.0 if is_normal(apath) else 1.0

            return nfeatures, nlabel, afeatures, alabel
        else:
            path = self.list[index]
            features = np.array(np.load(path, allow_pickle=True), dtype=np.float32)
            label = 0.0 if is_normal(path) else 1.0
            return features, label

    def __len__(self):
        if self.test_mode:
            return len(self.list)
        # train은 정상/비정상 쌍으로 묶으므로 둘 중 작은 길이
        return min(len(self.a_all), len(self.n_all))
