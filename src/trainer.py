from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from src.atlasnet import AtlasNetConfig, AtlasnetDecoder
from src.pointnet import PointNetConfig, PointNetEncoder
from src.utils import champfer_loss, generate_mesh_faces, transform_regularizer


@dataclass
class TrainerConfig:
    batch_size: int = 16
    lr: float = 1e-4
    n_epochs: int = 100
    log_dir: str = "runs/atlasnet"
    device: str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    save_dir: str = "checkpoints"
    log_img_interval: int = 2
    lambda_transform: float = 0.001


@dataclass
class TrainerState:
    epoch: int = 0
    global_step: int = 0
    best_loss: float = float("inf")


class AtlasNetTrainer:
    def __init__(self, config: TrainerConfig) -> None:
        self.cfg = config
        self.device = torch.device(config.device)
        self.writer = SummaryWriter(log_dir=config.log_dir)
        self.state = TrainerState()
        self._has_logged_graph = False
        self._histogram_interval = 50

        self.encoder = PointNetEncoder(PointNetConfig()).to(self.device)
        self.decoder = AtlasnetDecoder(AtlasNetConfig()).to(self.device)

        parameters = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = optim.Adam(parameters, lr=config.lr)

        os.makedirs(config.save_dir, exist_ok=True)
        self._log_initial_metadata(parameters)

    def train(self, dataloader: DataLoader) -> None:
        dataset_size = len(getattr(dataloader, "dataset", []))
        self._histogram_interval = max(1, len(dataloader) // 4 or 1)
        start_time = time.perf_counter()
        for epoch in range(1, self.cfg.n_epochs + 1):
            self.state.epoch = epoch
            self.encoder.train()
            self.decoder.train()

            epoch_loss = 0.0
            epoch_precision = 0.0
            epoch_recall = 0.0

            progress = tqdm(
                enumerate(dataloader),
                total=len(dataloader),
                desc=f"Epoch {epoch}/{self.cfg.n_epochs}",
                leave=False,
            )

            last_batch: Tuple[Tensor, Tensor, Tensor] | None = None
            for _, batch in progress:
                batch_start = time.perf_counter()
                if isinstance(batch, (tuple, list)):
                    batch = batch[0]
                x_gt = batch.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                latent, trans_feat = self.encoder(x_gt)
                
                y_pred = self.decoder(latent)

                if not self._has_logged_graph:
                    self._log_computational_graphs(x_gt, latent)

                reconstruction = y_pred.reshape(x_gt.size(0), -1, 3)

                loss_dict = champfer_loss(reconstruction, x_gt)
                loss = loss_dict.total +\
                    self.cfg.lambda_transform * transform_regularizer(trans_feat)
                loss.backward()

                grad_norm = self._compute_gradient_norm()
                param_norm = self._compute_parameter_norm()

                self.optimizer.step()

                batch_time = time.perf_counter() - batch_start
                throughput = x_gt.size(0) / max(batch_time, 1e-9)
                self._log_step_metrics(
                    loss_dict=loss_dict,
                    param_norm=param_norm,
                    grad_norm=grad_norm,
                    latent=latent.detach(),
                    reconstruction=reconstruction.detach(),
                    decoder_output=y_pred.detach(),
                    trans_feat=trans_feat.detach() if trans_feat is not None else None,
                    throughput=throughput,
                    batch_time=batch_time,
                )

                self.state.global_step += 1
                epoch_loss += loss.item()
                epoch_precision += loss_dict.precision.item()
                epoch_recall += loss_dict.recall.item()

                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    precision=f"{loss_dict.precision.item():.4f}",
                    recall=f"{loss_dict.recall.item():.4f}",
                    grad=f"{grad_norm:.2f}",
                    lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                    trans_reg=f"{self.cfg.lambda_transform * transform_regularizer(trans_feat).item():.4f}",
                )

                last_batch = (
                    x_gt.detach().cpu(),
                    reconstruction.detach().cpu(),
                    y_pred.detach().cpu(),
                )

            avg_loss = epoch_loss / len(dataloader)
            avg_precision = epoch_precision / len(dataloader)
            avg_recall = epoch_recall / len(dataloader)

            self._log_epoch_metrics(avg_loss, avg_precision, avg_recall)
            self._log_parameter_distributions(epoch)
            self._update_best_checkpoint(avg_loss)

            if epoch % self.cfg.log_img_interval == 0 and last_batch is not None:
                self._log_reconstructions(last_batch[0], last_batch[1], last_batch[2], epoch)

            if epoch % 10 == 0:
                self._save_checkpoint(epoch)

            elapsed = time.perf_counter() - start_time
            wall_clock = elapsed / epoch
            self.writer.add_scalar("train/epoch_time", wall_clock, epoch)
            if dataset_size:
                self.writer.add_scalar("train/samples_per_second", dataset_size / max(wall_clock, 1e-9), epoch)
                self.writer.add_scalar("train/epochs_per_hour", 3600.0 / max(wall_clock, 1e-9), epoch)

        self.writer.flush()
        self.writer.close()

    def _log_initial_metadata(self, parameters: Iterable[nn.Parameter]) -> None:
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        total_params = sum(p.numel() for p in parameters)
        trainable_params = sum(p.numel() for p in parameters if p.requires_grad)

        metadata = {
            "trainer": asdict(self.cfg),
            "encoder_params": encoder_params,
            "decoder_params": decoder_params,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "device": str(self.device),
        }

        self.writer.add_text("metadata/config", json.dumps(metadata, indent=2))
        self.writer.add_custom_scalars(
            {
                "Losses": {
                    "Chamfer": ["Multiline", ["train/loss_step", "train/loss_epoch"]],
                    "Components": ["Multiline", ["train/loss_precision", "train/loss_recall"]],
                },
                "Optimization": {
                    "Norms": ["Multiline", ["train/param_norm", "train/grad_norm"]],
                    "Speed": ["Multiline", ["train/throughput", "train/batch_time"]],
                },
            }
        )

    def _log_computational_graphs(self, x_gt: Tensor, latent: Tensor) -> None:
        with torch.inference_mode():
            self.writer.add_graph(self.encoder, x_gt)
            self.writer.add_graph(self.decoder, latent)
            class _AtlasNetPipeline(nn.Module):
                def __init__(self, encoder: PointNetEncoder, decoder: AtlasnetDecoder) -> None:
                    super().__init__()
                    self.encoder = encoder
                    self.decoder = decoder

                def forward(self, inputs: Tensor) -> Tensor:
                    features, _ = self.encoder(inputs)
                    return self.decoder(features)

            pipeline = _AtlasNetPipeline(self.encoder, self.decoder)
            self.writer.add_graph(pipeline, x_gt)
        self._has_logged_graph = True

    def _log_step_metrics(
        self,
        *,
        loss_dict,
        param_norm: float,
        grad_norm: float,
        latent: Tensor,
        reconstruction: Tensor,
        decoder_output: Tensor,
        trans_feat: Tensor | None,
        throughput: float,
        batch_time: float,
    ) -> None:
        step = self.state.global_step
        self.writer.add_scalar("train/loss_step", loss_dict.total.item(), step)
        self.writer.add_scalar("train/loss_precision", loss_dict.precision.item(), step)
        self.writer.add_scalar("train/loss_recall", loss_dict.recall.item(), step)
        self.writer.add_scalar("train/param_norm", param_norm, step)
        self.writer.add_scalar("train/grad_norm", grad_norm, step)
        self.writer.add_scalar("train/throughput", throughput, step)
        self.writer.add_scalar("train/batch_time", batch_time, step)
        self.writer.add_scalar("train/learning_rate", self.optimizer.param_groups[0]["lr"], step)

        if step % self._histogram_interval == 0:
            self.writer.add_histogram("latent/activations", latent.cpu(), step)
            self.writer.add_histogram("reconstruction/points", reconstruction.cpu(), step)
            self.writer.add_histogram("decoder/patches", decoder_output.cpu(), step)
            if trans_feat is not None:
                self.writer.add_histogram("transforms/feature", trans_feat.cpu(), step)

    def _log_epoch_metrics(self, loss: float, precision: float, recall: float) -> None:
        epoch = self.state.epoch
        self.writer.add_scalar("train/loss_epoch", loss, epoch)
        self.writer.add_scalar("train/precision_epoch", precision, epoch)
        self.writer.add_scalar("train/recall_epoch", recall, epoch)
        self.writer.add_scalar("train/best_loss", self.state.best_loss, epoch)

    def _log_parameter_distributions(self, epoch: int) -> None:
        for name, param in self.encoder.named_parameters():
            self.writer.add_histogram(f"encoder/params/{name}", param.detach().cpu(), epoch)
            if param.grad is not None:
                self.writer.add_histogram(f"encoder/grads/{name}", param.grad.detach().cpu(), epoch)
        for name, param in self.decoder.named_parameters():
            self.writer.add_histogram(f"decoder/params/{name}", param.detach().cpu(), epoch)
            if param.grad is not None:
                self.writer.add_histogram(f"decoder/grads/{name}", param.grad.detach().cpu(), epoch)

    def _log_reconstructions(self, x_gt: Tensor, reconstruction: Tensor, decoder_output: Tensor, epoch: int) -> None:
        self.encoder.eval()
        self.decoder.eval()

        x_gt_np = x_gt[0].numpy()
        y_pred_np = reconstruction[0].numpy()

        fig_pc = plt.figure(figsize=(8, 4))
        ax1 = fig_pc.add_subplot(121, projection="3d")
        ax1.scatter(x_gt_np[:, 0], x_gt_np[:, 1], x_gt_np[:, 2], c="b", s=2)
        ax1.set_title("Ground Truth")
        ax2 = fig_pc.add_subplot(122, projection="3d")
        ax2.scatter(y_pred_np[:, 0], y_pred_np[:, 1], y_pred_np[:, 2], c="r", s=2)
        ax2.set_title("Reconstruction")
        self.writer.add_figure("reconstruction/pointcloud", fig_pc, epoch)
        plt.close(fig_pc)

        points_per_patch = decoder_output.size(2)
        resolution = max(1, round(points_per_patch**0.5))
        faces = generate_mesh_faces(resolution, resolution, self.decoder.config.k_patches, device="cpu")
        mesh_vertices = torch.from_numpy(y_pred_np).unsqueeze(0).float()
        self.writer.add_mesh(
            "reconstruction/mesh",
            mesh_vertices,
            faces=faces.unsqueeze(0),
            global_step=epoch,
        )
        self.writer.add_histogram("mesh/vertex_distribution", mesh_vertices, epoch)

    def _update_best_checkpoint(self, loss: float) -> None:
        if loss < self.state.best_loss:
            self.state.best_loss = loss
            best_path = os.path.join(self.cfg.save_dir, "atlasnet_best.pth")
            torch.save(
                {
                    "encoder": self.encoder.state_dict(),
                    "decoder": self.decoder.state_dict(),
                    "epoch": self.state.epoch,
                    "global_step": self.state.global_step,
                    "best_loss": loss,
                },
                best_path,
            )

    def _save_checkpoint(self, epoch: int) -> None:
        path = os.path.join(self.cfg.save_dir, f"atlasnet_epoch{epoch}.pth")
        torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "decoder": self.decoder.state_dict(),
                "epoch": epoch,
                "global_step": self.state.global_step,
            },
            path,
        )

    def _compute_gradient_norm(self) -> float:
        norms = [
            (param.grad.detach().norm(2) ** 2)
            for param in self.encoder.parameters()
            if param.grad is not None
        ]
        norms.extend(
            (param.grad.detach().norm(2) ** 2)
            for param in self.decoder.parameters()
            if param.grad is not None
        )
        if not norms:
            return 0.0
        return torch.sqrt(torch.stack(norms).sum()).item()

    def _compute_parameter_norm(self) -> float:
        norms = [(param.detach().norm(2) ** 2) for param in self.encoder.parameters()]
        norms.extend((param.detach().norm(2) ** 2) for param in self.decoder.parameters())
        return torch.sqrt(torch.stack(norms).sum()).item()


if __name__ == "__main__":
    class DummyShapeNet(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 64

        def __getitem__(self, idx: int) -> Tensor:
            return torch.rand(2500, 3)

    dataset = DummyShapeNet()
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

    trainer = AtlasNetTrainer(TrainerConfig(n_epochs=2, log_dir="runs/atlasnet_dummy"))
    trainer.train(dataloader)
