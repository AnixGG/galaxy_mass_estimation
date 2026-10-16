"""Загрузка карт ACT+Planck, красных смещений и логарифмов масс скоплений."""

import torch
from torch.utils.data import Dataset
import numpy as np
import os


class ACTClusterDataset(Dataset):
    """Возвращает изображение, красное смещение и целевую logM500cUncorr.

    ``augm`` и ``norm`` выбирают подготовленные на диске варианты данных.
    Преобразования изображений при загрузке не выполняются.
    """

    def __init__(self, root_dir, indices, augm=False, norm=False):
        self.root_dir = root_dir
        self.indices = indices
        self.augm = augm
        self.norm = norm

        if not augm:
            self.redshifts = np.load(os.path.join(root_dir, "redshifts.npy"))
            self.massa = np.load(os.path.join(root_dir, "logM500cUncorr.npy"))
        else:
            self.redshifts = np.load(os.path.join(root_dir, "redshifts_augm.npy"))
            self.massa = np.load(os.path.join(root_dir, "logM500cUncorr_augm.npy"))
        self.n_samples = len(indices)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        idx = self.indices[idx]

        if not self.augm and not self.norm:
            img_raw = np.load(os.path.join(self.root_dir, f"images/{idx}.npy"))
        elif not self.augm and self.norm:
            img_raw = np.load(
                os.path.join(self.root_dir, f"images_normalized/{idx}.npy")
            )
        elif not self.norm:
            img_raw = np.load(os.path.join(self.root_dir, f"images_augm/{idx}.npy"))
        else:
            if idx <= 9976:
                img_raw = np.load(
                    os.path.join(self.root_dir, f"images_normalized/{idx}.npy")
                )
            else:
                img_raw = np.load(
                    os.path.join(self.root_dir, f"images_norm_augm/{idx+1}.npy")
                )

        img = torch.from_numpy(img_raw.astype(np.float32))

        scalars = torch.tensor(
            [
                self.redshifts[idx],
            ],
            dtype=torch.float32,
        )

        target = torch.tensor(self.massa[idx], dtype=torch.float32)
        return img, scalars, target
