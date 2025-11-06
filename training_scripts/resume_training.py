from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.data import DatasetAE, atlasnet_collate_fn
from src.trainer import AtlasNetTrainerAE, TrainerConfig
from src.utils import chamfer_loss

data_dir = 'data'
output_dir = 'outputs/ae'


ckpt_dir = Path(output_dir) / "checkpoints"
latest = sorted(ckpt_dir.glob("atlasnet_epoch*.pth"))[-1]

seed = 42
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
  torch.cuda.manual_seed_all(seed)


dataset = DatasetAE(Path(data_dir).expanduser().resolve(), num_points=None, cache=False)
indices = list(range(len(dataset)))
random.shuffle(indices)
val_count = max(1, int(round(len(dataset) * 0.1)))
val_indices = indices[:val_count]
train_indices = indices[val_count:]

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)

augment_params = {"jitter_std": 0.02, "jitter_clip": 0.05, "scale_min": 2.0 / 3.0, "scale_max": 3.0 / 2.0}
collate_train = atlasnet_collate_fn(2500, augment=True, augment_params=augment_params)
collate_val   = atlasnet_collate_fn(2500, augment=False)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=4,
                          pin_memory=torch.cuda.is_available(), collate_fn=collate_train)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=4,
                          pin_memory=torch.cuda.is_available(), collate_fn=collate_val)

device = "cuda" if torch.cuda.is_available() else "cpu"

trainer = AtlasNetTrainerAE.from_checkpoint(
    latest,
    overrides={
        "device": device,
        "n_epochs": 400,
        "log_dir": str(Path(output_dir) / "runs_resume"),
        "save_dir": str(Path(output_dir) / "checkpoints_resume"),
    },
    map_location=device,
    load_optimizer=True,
)

trainer.train(train_loader, val_loader)
