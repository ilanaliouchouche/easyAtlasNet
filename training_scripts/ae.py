from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.data import DatasetAE, atlasnet_collate_fn
from src.trainer import AtlasNetTrainerAE, TrainerConfig
from src.utils import champfer_loss


def evaluate_autoencoder(trainer: AtlasNetTrainerAE, dataloader: DataLoader) -> dict[str, float]:
    device = trainer.device
    trainer.encoder.eval()
    trainer.decoder.eval()
    total_loss = 0.0
    total_precision = 0.0
    total_recall = 0.0
    steps = 0
    with torch.inference_mode():
        for batch in dataloader:
            points = batch.to(device)
            latent, _ = trainer.encoder(points)
            output = trainer.decoder(latent)
            reconstruction = output.reshape(points.size(0), -1, 3)
            loss_dict = champfer_loss(reconstruction, points)
            total_loss += loss_dict.total.item()
            total_precision += loss_dict.precision.item()
            total_recall += loss_dict.recall.item()
            steps += 1
    if steps == 0:
        return {"loss": 0.0, "precision": 0.0, "recall": 0.0}
    return {
        "loss": total_loss / steps,
        "precision": total_precision / steps,
        "recall": total_recall / steps,
    }


def train_autoencoder(
    data_root: str,
    *,
    output_dir: str = "outputs/ae",
    batch_size: int = 16,
    num_epochs: int = 200,
    num_workers: int = 4,
    seed: int = 42,
) -> dict[str, float]:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    data_path = Path(data_root).expanduser().resolve()
    dataset = DatasetAE(data_path, num_points=None, cache=False)
    if len(dataset) < 2:
        raise ValueError("Dataset requires at least two samples")
    val_count = max(1, int(round(len(dataset) * 0.1)))
    train_count = len(dataset) - val_count
    if train_count <= 0:
        raise ValueError("Not enough samples for training split")
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    val_indices = indices[:val_count]
    train_indices = indices[val_count:]
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    augment_params = {"jitter_std": 0.02, "jitter_clip": 0.05, "scale_min": 2.0 / 3.0, "scale_max": 3.0 / 2.0}
    collate_train = atlasnet_collate_fn(2500, augment=False, augment_params=augment_params)
    collate_val = atlasnet_collate_fn(2500, augment=False)
    pin_memory = torch.cuda.is_available() or torch.backends.mps.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_train,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_val,
        drop_last=False,
    )
    steps_per_epoch = max(1, len(train_loader))
    metric_interval = max(1, steps_per_epoch // 10)
    image_interval = max(1, steps_per_epoch // 2)
    base_dir = Path(output_dir).expanduser().resolve()
    log_dir = base_dir / "runs"
    checkpoint_dir = base_dir / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config = TrainerConfig(
        batch_size=batch_size,
        lr=1e-4,
        n_epochs=num_epochs,
        log_dir=str(log_dir),
        save_dir=str(checkpoint_dir),
        log_val_step_interval=metric_interval,
        log_image_step_interval=image_interval,
        encoder_options={"trainable": True},
        decoder_options={"k_patches": 50, "total_n_points": 2500},
        scheduler_config={"type": "StepLR", "step_size": 40, "gamma": 0.1},
    )
    trainer = AtlasNetTrainerAE(config)
    trainer.train(train_loader, val_loader)
    metrics = evaluate_autoencoder(trainer, val_loader)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs/ae")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_autoencoder(
        args.data_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    print(
        f"Validation loss: {metrics['loss']:.6f}, "
        f"precision: {metrics['precision']:.6f}, "
        f"recall: {metrics['recall']:.6f}"
    )


if __name__ == "__main__":
    main()
