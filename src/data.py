from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

__all__ = ["DatasetAE", "atlasnet_collate_fn"]


def _gather_point_files(root: Path) -> list[Path]:
    paths: list[Path] = []
    queue: list[Path] = [root]
    while queue:
        current = queue.pop()
        for entry in current.iterdir():
            if entry.is_dir():
                queue.append(entry)
            elif entry.suffix.lower() == ".pts" and entry.parent.name == "points":
                paths.append(entry)
    paths.sort()
    return paths


def _load_pts(path: Path) -> np.ndarray:
    data = np.loadtxt(path, dtype=np.float32)
    if data.ndim == 1:
        data = np.expand_dims(data, 0)
    if data.shape[1] > 3:
        data = data[:, :3]
    return data.astype(np.float32, copy=False)


def _normalise(points: np.ndarray) -> np.ndarray:
    centre = points.mean(axis=0, keepdims=True)
    points = points - centre
    scale = np.linalg.norm(points, axis=1).max()
    if scale > 0:
        points = points / scale
    return points


def _resample(points: np.ndarray, n_points: int, seed: int) -> np.ndarray:
    if points.shape[0] == n_points:
        return points
    rng = np.random.default_rng(seed)
    replace = points.shape[0] < n_points
    indices = rng.choice(points.shape[0], size=n_points, replace=replace)
    return points[indices]


class DatasetAE(Dataset):
    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        num_points: int | None = None,
        normalize: bool = True,
        cache: bool = False,
        transform: Callable[[Tensor], Tensor] | None = None,
        files: Sequence[str | os.PathLike[str]] | None = None,
        seed: int | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        if files is not None:
            self.files = [Path(f).expanduser().resolve() for f in files]
        else:
            if not self.root.is_dir():
                raise FileNotFoundError(self.root)
            self.files = _gather_point_files(self.root)
        if not self.files:
            raise ValueError("No .pts files found")
        self.num_points = num_points
        self.normalize = normalize
        self.cache = cache
        self.transform = transform
        self.seed = 0 if seed is None else seed
        self._cache_data: dict[int, Tensor] = {}

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Tensor:
        if self.cache and index in self._cache_data:
            points = self._cache_data[index]
        else:
            points_np = _load_pts(self.files[index])
            if self.normalize:
                points_np = _normalise(points_np)
            if self.num_points is not None:
                points_np = _resample(points_np, self.num_points, self.seed + index)
            points = torch.from_numpy(points_np.copy())
            if self.cache:
                self._cache_data[index] = points
        if self.transform is not None:
            src = points.clone() if self.cache else points
            points = self.transform(src)
        return points


def _normalize_batch(points: Tensor) -> Tensor:
    centre = points.mean(dim=1, keepdim=True)
    points = points - centre
    norms = points.norm(dim=2).max(dim=1, keepdim=True).values.clamp(min=1e-12)
    return points / norms.unsqueeze(-1)


def _augment_batch(points: Tensor, params: dict[str, float]) -> Tensor:
    jitter_std = params.get("jitter_std", 0.02)
    jitter_clip = params.get("jitter_clip", 0.05)
    scale_min = params.get("scale_min", 2.0 / 3.0)
    scale_max = params.get("scale_max", 3.0 / 2.0)

    device = points.device
    batch_size, n_points, _ = points.shape

    theta = torch.rand(batch_size, device=device) * 2 * torch.pi
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    rot = torch.stack(
        [
            torch.stack([cos_theta, -sin_theta, torch.zeros_like(theta)], dim=1),
            torch.stack([sin_theta, cos_theta, torch.zeros_like(theta)], dim=1),
            torch.stack([torch.zeros_like(theta), torch.zeros_like(theta), torch.ones_like(theta)], dim=1),
        ],
        dim=1,
    )
    points = torch.bmm(points, rot.transpose(1, 2))

    scales = torch.empty(batch_size, 1, 1, device=device).uniform_(scale_min, scale_max)
    points = points * scales

    jitter = torch.randn(batch_size, n_points, 3, device=device) * jitter_std
    jitter = torch.clamp(jitter, -jitter_clip, jitter_clip)
    points = points + jitter

    return _normalize_batch(points)


def _atlasnet_collate(num_points: int, augment: bool, params: dict[str, float], batch: Sequence[Tensor]) -> Tensor:
    out = []
    for item in batch:
        points = item
        if points.dim() != 2:
            points = points.view(-1, 3)
        count = points.size(0)
        if count == num_points:
            out.append(points)
            continue
        if count > num_points:
            idx = torch.randperm(count)[:num_points]
            out.append(points[idx])
            continue
        idx = torch.randint(0, count, (num_points,))
        out.append(points[idx])
    stacked = torch.stack(out, dim=0)
    if augment:
        stacked = _augment_batch(stacked, params)
    return stacked


def atlasnet_collate_fn(
    num_points: int = 2500,
    *,
    augment: bool = False,
    augment_params: dict[str, float] | None = None,
) -> Callable[[Sequence[Tensor]], Tensor]:
    params = augment_params or {}
    return functools.partial(_atlasnet_collate, num_points, augment, params)


if __name__ == "__main__":
    import shutil

    tmp = Path("tmp_pts")
    pts_dir = tmp / "cat" / "points"
    pts_dir.mkdir(parents=True, exist_ok=True)
    data = np.random.randn(1000, 3).astype(np.float32)
    np.savetxt(pts_dir / "sample.pts", data)
    dataset = DatasetAE(tmp, num_points=512, seed=42, cache=True)
    sample = dataset[0]
    loader = atlasnet_collate_fn(256)
    batch = loader([sample, sample])
    print(sample.shape, batch.shape)
    shutil.rmtree(tmp)
