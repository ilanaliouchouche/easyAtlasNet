from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, fields, is_dataclass
from typing import Any, Iterable, Tuple

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import ResNet18_Weights, resnet18
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
    device: str = ("cuda" if torch.cuda.is_available()
                   else ("mps" if torch.backends.mps.is_available() else "cpu"))
    save_dir: str = "checkpoints"
    log_val_step_interval: int = 1
    log_image_step_interval: int = 1000
    lambda_transform: float = 0.001
    encoder_options: dict[str, Any] | None = None
    decoder_options: dict[str, Any] | None = None
    scheduler_config: dict[str, Any] | None = None


@dataclass
class TrainerState:
    epoch: int = 0
    global_step: int = 0
    val_global_step: int = 0
    best_loss: float = float("inf")


class AtlasNetTrainer(ABC):
    def __init__(self, config: TrainerConfig) -> None:
        self.cfg = config
        self.device = torch.device(config.device)
        self.writer = SummaryWriter(log_dir=config.log_dir)
        self.state = TrainerState()
        self._has_logged_graph = False
        self._histogram_interval = 50

        self._encoder_options = dict(config.encoder_options) if isinstance(config.encoder_options, dict) else {}
        self._decoder_options = config.decoder_options if config.decoder_options is not None else {}
        self.decoder_config = self._build_decoder_config()
        self.decoder = self._build_decoder().to(self.device)
        self.encoder = self._build_encoder().to(self.device)
        self.cfg.encoder_options = dict(self._encoder_options)
        self.cfg.decoder_options = asdict(self.decoder.config)

        parameters = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = optim.Adam(parameters, lr=config.lr)
        self.scheduler = self._build_scheduler()

        os.makedirs(config.save_dir, exist_ok=True)
        self._log_initial_metadata(parameters)

    def train(self, train_loader: DataLoader, val_loader: DataLoader | None = None) -> None:
        dataset_size = len(getattr(train_loader, "dataset", []))
        base_hist = len(train_loader) // 4 if len(train_loader) else 0
        self._histogram_interval = max(1, base_hist)
        start_time = time.perf_counter()
        for epoch in range(1, self.cfg.n_epochs + 1):
            self.state.epoch = epoch
            self.encoder.train()
            self.decoder.train()

            epoch_loss = 0.0
            epoch_precision = 0.0
            epoch_recall = 0.0

            progress = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Epoch {epoch}/{self.cfg.n_epochs}",
                leave=False,
            )

            for _, batch in progress:
                batch_start = time.perf_counter()
                inputs, x_gt = self._prepare_batch(batch)
                inputs = inputs.to(self.device)
                x_gt = x_gt.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                latent, aux = self.encoder(inputs)

                y_pred = self.decoder(latent)

                if not self._has_logged_graph:
                    self._log_computational_graphs(inputs)

                reconstruction = y_pred.reshape(x_gt.size(0), -1, 3)

                loss_dict = champfer_loss(reconstruction, x_gt)
                reg_loss = self._regularization(aux)
                loss = loss_dict.total + reg_loss
                loss.backward()

                grad_norm = self._compute_gradient_norm()
                param_norm = self._compute_parameter_norm()

                self.optimizer.step()

                batch_time = time.perf_counter() - batch_start
                throughput = x_gt.size(0) / max(batch_time, 1e-9)
                self.state.global_step += 1
                self._log_step_metrics(
                    loss_dict=loss_dict,
                    param_norm=param_norm,
                    grad_norm=grad_norm,
                    latent=latent.detach(),
                    reconstruction=reconstruction.detach(),
                    decoder_output=y_pred.detach(),
                    aux=aux.detach() if isinstance(aux, Tensor) else None,
                    throughput=throughput,
                    batch_time=batch_time,
                    reg_loss=reg_loss.detach(),
                )

                if (
                    self.cfg.log_image_step_interval > 0
                    and self.state.global_step % self.cfg.log_image_step_interval == 0
                ):
                    self._log_reconstructions(
                        x_gt.detach().cpu(),
                        reconstruction.detach().cpu(),
                        y_pred.detach().cpu(),
                        self.state.global_step,
                    )
                epoch_loss += loss.item()
                epoch_precision += loss_dict.precision.item()
                epoch_recall += loss_dict.recall.item()

                progress.set_postfix(
                    loss=f"{loss.item():.4f}",
                    precision=f"{loss_dict.precision.item():.4f}",
                    recall=f"{loss_dict.recall.item():.4f}",
                    grad=f"{grad_norm:.2f}",
                    lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                    regularizer=f"{reg_loss.item():.4f}",
                )

            avg_loss = epoch_loss / len(train_loader)
            avg_precision = epoch_precision / len(train_loader)
            avg_recall = epoch_recall / len(train_loader)

            self._log_epoch_metrics(avg_loss, avg_precision, avg_recall)
            self._log_parameter_distributions(epoch)
            self._update_best_checkpoint(avg_loss)

            if epoch % 10 == 0:
                self._save_checkpoint(epoch)

            if val_loader is not None:
                self._validate(val_loader)

            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                self.writer.add_scalar("train/lr_epoch", current_lr, epoch)

            elapsed = time.perf_counter() - start_time
            wall_clock = elapsed / epoch
            self.writer.add_scalar("train/epoch_time", wall_clock, epoch)
            if dataset_size:
                self.writer.add_scalar("train/samples_per_second", dataset_size / max(wall_clock, 1e-9), epoch)
                self.writer.add_scalar("train/epochs_per_hour", 3600.0 / max(wall_clock, 1e-9), epoch)

        self.writer.flush()
        self.writer.close()

    @abstractmethod
    def _prepare_batch(self, batch: Any) -> Tuple[Tensor, Tensor]:
        ...

    def _build_decoder_config(self) -> AtlasNetConfig:
        if isinstance(self._decoder_options, AtlasNetConfig):
            return self._decoder_options
        if isinstance(self._decoder_options, dict) and self._decoder_options:
            return AtlasNetConfig(**self._decoder_options)
        return AtlasNetConfig()

    def _build_decoder(self) -> AtlasnetDecoder:
        return AtlasnetDecoder(self.decoder_config)

    @abstractmethod
    def _build_encoder(self) -> nn.Module:
        ...

    def _regularization(self, aux: Any) -> Tensor:
        return torch.zeros((), device=self.device)

    def _build_scheduler(self):
        cfg = self.cfg.scheduler_config
        if not cfg:
            return None
        scheduler_type = str(cfg.get("type", "")).lower()
        if scheduler_type == "steplr":
            step_size = cfg.get("step_size", 40)
            gamma = cfg.get("gamma", 0.1)
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        if scheduler_type == "multisteplr":
            milestones = cfg.get("milestones")
            gamma = cfg.get("gamma", 0.1)
            if milestones is None:
                raise ValueError("MultiStepLR requires 'milestones'.")
            return optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)
        if scheduler_type == "cosineannealinglr":
            t_max = cfg.get("t_max", max(1, self.cfg.n_epochs))
            eta_min = cfg.get("eta_min", 0.0)
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=t_max, eta_min=eta_min)
        raise ValueError(f"Unsupported scheduler type '{scheduler_type}'.")

    def _serialize_for_logging(self, value: Any) -> Any:
        if is_dataclass(value):
            return self._serialize_for_logging(asdict(value))
        if isinstance(value, dict):
            return {k: self._serialize_for_logging(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._serialize_for_logging(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def _log_initial_metadata(self, parameters: Iterable[nn.Parameter]) -> None:
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        total_params = sum(p.numel() for p in parameters)
        trainable_params = sum(p.numel() for p in parameters if p.requires_grad)

        metadata = {
            "trainer": self._serialize_for_logging(asdict(self.cfg)),
            "encoder_params": encoder_params,
            "decoder_params": decoder_params,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "device": str(self.device),
            "scheduler": self._serialize_for_logging(self.cfg.scheduler_config),
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

    def _log_computational_graphs(self, inputs: Tensor) -> None:
        with torch.inference_mode():
            class _AtlasNetPipeline(nn.Module):
                def __init__(self, encoder: nn.Module, decoder: AtlasnetDecoder) -> None:
                    super().__init__()
                    self.encoder = encoder
                    self.decoder = decoder

                def forward(self, x: Tensor) -> Tensor:
                    features, _ = self.encoder(x)
                    return self.decoder(features)

            pipeline = _AtlasNetPipeline(self.encoder, self.decoder)
            self.writer.add_graph(pipeline, inputs)
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
        aux: Tensor | None,
        throughput: float,
        batch_time: float,
        reg_loss: Tensor,
    ) -> None:
        step = self.state.global_step
        self.writer.add_scalar("train/loss_step", loss_dict.total.item(), step)
        self.writer.add_scalar("train/loss_precision", loss_dict.precision.item(), step)
        self.writer.add_scalar("train/loss_recall", loss_dict.recall.item(), step)
        self.writer.add_scalar("train/param_norm", param_norm, step)
        self.writer.add_scalar("train/grad_norm", grad_norm, step)
        self.writer.add_scalar("train/throughput", throughput, step)
        self.writer.add_scalar("train/batch_time", batch_time, step)
        self.writer.add_scalar("train/regularizer", reg_loss.item(), step)
        self.writer.add_scalar("train/learning_rate", self.optimizer.param_groups[0]["lr"], step)

        if step % self._histogram_interval == 0:
            self.writer.add_histogram("latent/activations", latent.cpu(), step)
            self.writer.add_histogram("reconstruction/points", reconstruction.cpu(), step)
            self.writer.add_histogram("decoder/patches", decoder_output.cpu(), step)
            if aux is not None and aux.numel() > 0:
                self.writer.add_histogram("auxiliary/features", aux.cpu(), step)

    def _log_validation_step_metrics(self, *, loss_dict, reg_loss: Tensor) -> None:
        step = self.state.val_global_step
        total = loss_dict.total.item() + reg_loss.item()
        self.writer.add_scalar("val/loss_step", total, step)
        self.writer.add_scalar("val/loss_precision", loss_dict.precision.item(), step)
        self.writer.add_scalar("val/loss_recall", loss_dict.recall.item(), step)
        self.writer.add_scalar("val/regularizer", reg_loss.item(), step)

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

    def _validate(self, dataloader: DataLoader) -> None:
        self.encoder.eval()
        self.decoder.eval()

        total_loss = 0.0
        total_precision = 0.0
        total_recall = 0.0
        steps = 0

        with torch.no_grad():
            for batch in dataloader:
                inputs, x_gt = self._prepare_batch(batch)
                inputs = inputs.to(self.device)
                x_gt = x_gt.to(self.device)

                latent, aux = self.encoder(inputs)
                y_pred = self.decoder(latent)
                reconstruction = y_pred.reshape(x_gt.size(0), -1, 3)

                loss_dict = champfer_loss(reconstruction, x_gt)
                reg_loss = self._regularization(aux)

                total_loss += loss_dict.total.item() + reg_loss.item()
                total_precision += loss_dict.precision.item()
                total_recall += loss_dict.recall.item()

                self.state.val_global_step += 1
                if (
                    self.cfg.log_val_step_interval > 0
                    and self.state.val_global_step % self.cfg.log_val_step_interval == 0
                ):
                    self._log_validation_step_metrics(
                        loss_dict=loss_dict,
                        reg_loss=reg_loss.detach(),
                    )

                steps += 1

        if steps == 0:
            return

        avg_loss = total_loss / steps
        avg_precision = total_precision / steps
        avg_recall = total_recall / steps
        epoch = self.state.epoch

        self.writer.add_scalar("val/loss_epoch", avg_loss, epoch)
        self.writer.add_scalar("val/precision_epoch", avg_precision, epoch)
        self.writer.add_scalar("val/recall_epoch", avg_recall, epoch)

    def _log_reconstructions(self, x_gt: Tensor, reconstruction: Tensor, decoder_output: Tensor, epoch: int) -> None:
        self.encoder.eval()
        self.decoder.eval()

        x_gt_np = x_gt[0].numpy()
        y_pred_np = reconstruction[0].numpy()

        points_per_patch = decoder_output.size(2)
        resolution = max(1, round(points_per_patch**0.5))
        faces = generate_mesh_faces(resolution, resolution, self.decoder.config.k_patches, device="cpu")

        fig_pc = plt.figure(figsize=(12, 4))
        ax1 = fig_pc.add_subplot(131, projection="3d")
        ax1.scatter(x_gt_np[:, 0], x_gt_np[:, 1], x_gt_np[:, 2], c="b", s=2)
        ax1.set_title("Ground Truth")
        ax2 = fig_pc.add_subplot(132, projection="3d")
        ax2.scatter(y_pred_np[:, 0], y_pred_np[:, 1], y_pred_np[:, 2], c="r", s=2)
        ax2.set_title("Reconstruction")

        ax3 = fig_pc.add_subplot(133, projection="3d")
        if faces.numel() > 0:
            faces_np = faces.numpy()
            tri_vertices = y_pred_np[faces_np]
            mesh = Poly3DCollection(tri_vertices, alpha=0.35, edgecolor="k")
            mesh.set_facecolor((1.0, 0.3, 0.3, 0.35))
            ax3.add_collection3d(mesh)
        ax3.scatter(y_pred_np[:, 0], y_pred_np[:, 1], y_pred_np[:, 2], c="r", s=1)
        ax3.set_title("Triangle Mesh")
        self.writer.add_figure("reconstruction/pointcloud", fig_pc, epoch)
        plt.close(fig_pc)

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
            self.save_checkpoint(best_path)

    def _save_checkpoint(self, epoch: int) -> None:
        path = os.path.join(self.cfg.save_dir, f"atlasnet_epoch{epoch}.pth")
        self.save_checkpoint(path)

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

    def _checkpoint_state(self, include_optimizer: bool) -> dict[str, Any]:
        self.cfg.encoder_options = dict(self._encoder_options)
        self.cfg.decoder_options = asdict(self.decoder.config)
        state = {
            "trainer_cfg": asdict(self.cfg),
            "trainer_state": asdict(self.state),
            "encoder_state": self.encoder.state_dict(),
            "decoder_state": self.decoder.state_dict(),
            "decoder_config": asdict(self.decoder.config),
        }
        if include_optimizer:
            state["optimizer_state"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            state["scheduler_state"] = self.scheduler.state_dict()
        return state

    def save_checkpoint(self, path: str, *, include_optimizer: bool = True) -> None:
        payload = self._checkpoint_state(include_optimizer)
        torch.save(payload, path)

    def _apply_checkpoint(self, checkpoint: dict[str, Any], *, strict: bool = True, load_optimizer: bool = True) -> None:
        self.encoder.load_state_dict(checkpoint["encoder_state"], strict=strict)
        self.decoder.load_state_dict(checkpoint["decoder_state"], strict=strict)
        optimizer_loaded = False
        if load_optimizer and "optimizer_state" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            optimizer_loaded = True
        trainer_state = checkpoint.get("trainer_state")
        if trainer_state is not None:
            state_kwargs = {
                field.name: trainer_state.get(field.name, getattr(self.state, field.name))
                for field in fields(TrainerState)
            }
            self.state = TrainerState(**state_kwargs)
        cfg_state = checkpoint.get("trainer_cfg")
        if cfg_state is not None:
            cfg_dict = cfg_state if isinstance(cfg_state, dict) else asdict(cfg_state)
            for field in fields(TrainerConfig):
                name = field.name
                if name in cfg_dict:
                    setattr(self.cfg, name, cfg_dict[name])
            self._encoder_options = dict(self.cfg.encoder_options or {})
            self._decoder_options = self.cfg.decoder_options if self.cfg.decoder_options is not None else {}
        decoder_cfg_state = checkpoint.get("decoder_config")
        if decoder_cfg_state is not None:
            if is_dataclass(decoder_cfg_state):
                decoder_cfg_state = asdict(decoder_cfg_state)
            self._decoder_options = decoder_cfg_state
        self.cfg.encoder_options = dict(self._encoder_options)
        self.cfg.decoder_options = asdict(self.decoder.config)
        self._decoder_options = self.cfg.decoder_options
        self.device = torch.device(self.cfg.device)
        self.encoder.to(self.device)
        self.decoder.to(self.device)
        if optimizer_loaded:
            for state in self.optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, Tensor):
                        state[key] = value.to(self.device)
        if self.scheduler is not None and "scheduler_state" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
            for group, base_lr in zip(self.optimizer.param_groups, self.scheduler.base_lrs):
                group["lr"] = base_lr
        else:
            for group in self.optimizer.param_groups:
                group["lr"] = self.cfg.lr

    def load_checkpoint(
        self,
        path: str,
        *,
        map_location: Any | None = None,
        strict: bool = True,
        load_optimizer: bool = True,
    ) -> None:
        checkpoint = torch.load(path, map_location=map_location)
        self._apply_checkpoint(checkpoint, strict=strict, load_optimizer=load_optimizer)

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        *,
        overrides: dict[str, Any] | None = None,
        map_location: Any | None = None,
        strict: bool = True,
        load_optimizer: bool = True,
    ) -> AtlasNetTrainer:
        checkpoint = torch.load(path, map_location=map_location)
        cfg_data = checkpoint.get("trainer_cfg", {})
        if is_dataclass(cfg_data):
            cfg_dict = asdict(cfg_data)
        elif isinstance(cfg_data, dict):
            cfg_dict = dict(cfg_data)
        else:
            cfg_dict = {}
        if overrides:
            cfg_dict.update(overrides)
        trainer = cls(TrainerConfig(**cfg_dict))
        trainer._apply_checkpoint(checkpoint, strict=strict, load_optimizer=load_optimizer)
        return trainer


class AtlasNetTrainerAE(AtlasNetTrainer):
    def _prepare_batch(self, batch: Any) -> Tuple[Tensor, Tensor]:
        if isinstance(batch, (tuple, list)):
            batch = batch[0]
        if not isinstance(batch, Tensor):
            batch = torch.as_tensor(batch, dtype=torch.float32)
        return batch, batch

    def _build_encoder(self) -> nn.Module:
        options = self._encoder_options
        config_source = options.get("config")
        if isinstance(config_source, dict):
            pointnet_cfg = PointNetConfig(**config_source)
        elif isinstance(config_source, PointNetConfig):
            pointnet_cfg = config_source
        else:
            config_kwargs = options.get("config_kwargs", {})
            pointnet_cfg = PointNetConfig(**config_kwargs)
        encoder = PointNetEncoder(pointnet_cfg)
        trainable = bool(options.get("trainable", True))
        if not trainable:
            for param in encoder.parameters():
                param.requires_grad = False
        state_dict_path = options.get("state_dict_path")
        if state_dict_path:
            state_dict = torch.load(state_dict_path, map_location="cpu")
            encoder.load_state_dict(state_dict)
        options["config"] = asdict(pointnet_cfg)
        options["trainable"] = trainable
        if state_dict_path:
            options["state_dict_path"] = state_dict_path
        elif "state_dict_path" in options:
            options.pop("state_dict_path")
        return encoder

    def _regularization(self, aux: Any) -> Tensor:
        if aux is None:
            return torch.zeros((), device=self.device)
        if isinstance(aux, Tensor) and aux.numel() == 0:
            return torch.zeros((), device=self.device)
        return self.cfg.lambda_transform * transform_regularizer(aux)


class ResNet18Encoder(nn.Module):
    def __init__(self, latent_dim: int, *, weights: str | ResNet18_Weights | None = None, trainable: bool = True) -> None:
        super().__init__()
        resolved_weights = None
        if isinstance(weights, str):
            if not hasattr(ResNet18_Weights, weights):
                raise ValueError(f"Unknown ResNet18 weights identifier '{weights}'")
            resolved_weights = getattr(ResNet18_Weights, weights)
        else:
            resolved_weights = weights
        self.model = resnet18(weights=resolved_weights)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, latent_dim)
        if not trainable:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        features = self.model(x)
        empty = features.new_zeros((0,))
        return features, empty


class AtlasNetTrainerSVR(AtlasNetTrainer):
    def _prepare_batch(self, batch: Any) -> Tuple[Tensor, Tensor]:
        if isinstance(batch, dict):
            inputs = batch["image"]
            target = batch["gt"]
        elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
            inputs, target = batch[0], batch[1]
        else:
            raise ValueError("Batch format not supported for SVR trainer.")
        if not isinstance(inputs, Tensor):
            inputs = torch.as_tensor(inputs, dtype=torch.float32)
        if not isinstance(target, Tensor):
            target = torch.as_tensor(target, dtype=torch.float32)
        return inputs, target

    def _build_encoder(self) -> nn.Module:
        options = self._encoder_options
        weights_option = options.get("weights")
        pretrained_option = options.get("pretrained")
        trainable = bool(options.get("trainable", True))
        if isinstance(weights_option, ResNet18_Weights):
            weight_name = weights_option.name
        elif isinstance(weights_option, str):
            weight_name = weights_option
        elif weights_option is True or pretrained_option:
            weight_name = ResNet18_Weights.DEFAULT.name
        else:
            weight_name = None
        encoder = ResNet18Encoder(
            self.decoder.config.latent_dim,
            weights=weight_name,
            trainable=trainable,
        )
        state_dict_path = options.get("state_dict_path")
        if state_dict_path:
            state_dict = torch.load(state_dict_path, map_location="cpu")
            encoder.load_state_dict(state_dict)
        if pretrained_option is None:
            pretrained_flag = bool(weight_name)
        else:
            pretrained_flag = bool(pretrained_option)
        options["weights"] = weight_name
        options["pretrained"] = pretrained_flag
        options["trainable"] = trainable
        if state_dict_path:
            options["state_dict_path"] = state_dict_path
        elif "state_dict_path" in options:
            options.pop("state_dict_path")
        return encoder


if __name__ == "__main__":
    class DummyShapeNet(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 64

        def __getitem__(self, idx: int) -> Tensor:
            return torch.rand(2500, 3)

    class DummySVRDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 64

        def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
            image = torch.rand(3, 224, 224)
            shape = torch.rand(2500, 3)
            return image, shape

    dataset_ae = DummyShapeNet()
    dataloader_ae = DataLoader(dataset_ae, batch_size=8, shuffle=True, num_workers=0)
    valloader_ae = DataLoader(dataset_ae, batch_size=8, shuffle=False, num_workers=0)
    trainer_ae = AtlasNetTrainerAE(TrainerConfig(n_epochs=1, log_dir="runs/atlasnet_dummy_ae"))
    trainer_ae.train(dataloader_ae, valloader_ae)

    dataset_svr = DummySVRDataset()
    dataloader_svr = DataLoader(dataset_svr, batch_size=4, shuffle=True, num_workers=0)
    valloader_svr = DataLoader(dataset_svr, batch_size=4, shuffle=False, num_workers=0)
    trainer_svr = AtlasNetTrainerSVR(TrainerConfig(n_epochs=1, log_dir="runs/atlasnet_dummy_svr"))
    trainer_svr.train(dataloader_svr, valloader_svr)
