from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable, List

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.data import DatasetAE, atlasnet_collate_fn
from src.trainer import AtlasNetTrainerAE, TrainerConfig
from src.utils import chamfer_loss


class _FixedBatchDataset(Dataset[Tensor]):

    def __init__(self, samples: Iterable[Tensor], repeats: int) -> None:
        samples_list: List[Tensor] = [s.clone() for s in samples]
        if not samples_list:
            raise ValueError("At least one sample is required to build the fixed batch dataset.")
        if repeats <= 0:
            raise ValueError("The number of repeats must be positive.")
        self._samples = samples_list
        self._repeats = repeats

    def __len__(self) -> int:
        return self._repeats

    def __getitem__(self, index: int) -> Tensor:
        return self._samples[index % len(self._samples)]


def _evaluate_on_loader(trainer: AtlasNetTrainerAE, dataloader: DataLoader) -> dict[str, float]:
    trainer.encoder.eval()
    trainer.decoder.eval()
    device = trainer.device
    total_loss = 0.0
    total_precision = 0.0
    total_recall = 0.0
    steps = 0
    with torch.inference_mode():
        for batch in dataloader:
            inputs, targets = trainer._prepare_batch(batch)
            inputs = inputs.to(device)
            targets = targets.to(device)
            latent, aux = trainer.encoder(inputs)
            preds = trainer.decoder(latent)
            reconstruction = preds.reshape(targets.size(0), -1, 3)
            loss_dict = chamfer_loss(reconstruction, targets)
            reg_loss = trainer._regularization(aux)
            total_loss += loss_dict.total.item() + reg_loss.item()
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train AtlasNet on a single mini-batch to check it can overfit (Karpathy sanity check)."
    )
    parser.add_argument("--data-root", type=str, default="data", help="Directory that contains the ShapeNet .pts files.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/overfit_one_batch",
        help="Directory used to save TensorBoard logs and checkpoints.",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=1,
        help="Number of distinct shapes to include in the fixed batch (typically 1).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for the sanity check. The dataset will repeat samples to match this size.",
    )
    parser.add_argument("--num-points", type=int, default=2500, help="Number of points per shape in the batch.")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs to train on the fixed batch.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate used by Adam.")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device to run on (cpu, cuda, mps). Defaults to cpu for stability on small tests.",
    )
    parser.add_argument(
        "--k-patches",
        type=int,
        default=25,
        help="Number of atlas patches. Keep it compatible with num-points when using regular grids.",
    )
    parser.add_argument(
        "--random-grid",
        action="store_true",
        help="Sample decoder patches from a random grid (avoids total_n_points divisibility constraints).",
    )
    parser.add_argument(
        "--lambda-transform",
        type=float,
        default=0.0,
        help="Weight of the STN regularisation term. Set to zero for easier overfitting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_root = Path(args.data_root).expanduser().resolve()
    dataset = DatasetAE(
        data_root,
        num_points=None,
        cache=True,
        normalize=True,
    )

    subset_count = min(args.subset_size, len(dataset))
    if subset_count <= 0:
        raise ValueError("Subset size must be at least 1 and the dataset cannot be empty.")

    samples = [dataset[idx] for idx in range(subset_count)]
    fixed_dataset = _FixedBatchDataset(samples, repeats=args.batch_size)

    collate = atlasnet_collate_fn(args.num_points, augment=False)
    pin_memory = torch.cuda.is_available() or torch.backends.mps.is_available()
    train_loader = DataLoader(
        fixed_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=collate,
        drop_last=False,
    )

    if len(train_loader) != 1:
        raise RuntimeError(
            f"Expected the fixed dataset to yield a single batch, but the dataloader returned {len(train_loader)} steps."
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    log_dir = output_dir / "runs"
    checkpoint_dir = output_dir / "checkpoints"
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    decoder_options = {
        "k_patches": args.k_patches,
        "total_n_points": args.num_points,
        "random_grid": args.random_grid,
    }

    config = TrainerConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        n_epochs=args.epochs,
        log_dir=str(log_dir),
        save_dir=str(checkpoint_dir),
        log_val_step_interval=0,
        log_image_step_interval=1,
        lambda_transform=args.lambda_transform,
        encoder_options={"trainable": True},
        decoder_options=decoder_options,
        scheduler_config=None,
        device=args.device,
        # reconstruction_loss="sliced_wasserstein",
    )

    trainer = AtlasNetTrainerAE(config)
    trainer.train(train_loader)

    metrics = _evaluate_on_loader(trainer, train_loader)
    print(
        f"[one-batch] loss: {metrics['loss']:.6f}, "
        f"precision: {metrics['precision']:.6f}, "
        f"recall: {metrics['recall']:.6f}"
    )
    print("If the loss is near zero, the model can overfit a single batch and the training loop is likely correct.")


if __name__ == "__main__":
    main()
