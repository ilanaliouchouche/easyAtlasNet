import numpy as np
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path

from src.trainer import AtlasNetTrainerAE
from src.data import atlasnet_collate_fn
from src.utils import generate_mesh_faces

checkpoint_path = Path("outputs/ae/checkpoints/atlasnet_epoch50.pth")
sample_path = sorted(Path("data/02691156/points").glob("*.pts"))[17]

points_np = np.loadtxt(sample_path, dtype=np.float32)
if points_np.ndim == 1:
    points_np = points_np[None, :]
points_np = points_np[:, :3]

collate = atlasnet_collate_fn(2500, augment=False)
points_tensor = torch.from_numpy(points_np)
collated = collate([points_tensor])  # (1, 2500, 3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trainer = AtlasNetTrainerAE.from_checkpoint(
    str(checkpoint_path),
    overrides={
        "device": str(device),
        "log_dir": "outputs/inference/runs",
        "save_dir": "outputs/inference/checkpoints",
        "n_epochs": 1,
    },
    map_location=device,
    load_optimizer=False,
)

trainer.encoder.eval()
trainer.decoder.eval()
with torch.inference_mode():
    latent, _ = trainer.encoder(collated.to(device))
    output = trainer.decoder(latent)
    reconstruction = output.reshape(-1, 3).cpu().numpy()

points_gt = collated.squeeze(0).numpy()
points_per_patch = reconstruction.shape[0] // trainer.decoder.config.k_patches
resolution = int(points_per_patch ** 0.5)
faces = generate_mesh_faces(resolution, resolution, trainer.decoder.config.k_patches).numpy()

print("Sample:", sample_path.name)
print("GT shape:", points_gt.shape)
print("Reconstruction shape:", reconstruction.shape)
print("Faces shape:", faces.shape)
np.set_printoptions(precision=6, suppress=True)
print("\nGround truth points (first 10):\n", points_gt[:10])
print("\nReconstruction points (first 10):\n", reconstruction[:10])
print("\nMesh faces (first 10):\n", faces[:10])

def _setup_axis(axis):
    axis.set_box_aspect([1, 1, 1])
    axis.grid(False)

fig_gt = plt.figure(figsize=(6, 6))
ax_gt = fig_gt.add_subplot(111, projection="3d")
ax_gt.scatter(
    points_gt[:, 0],
    points_gt[:, 1],
    points_gt[:, 2],
    s=4,
    c="tab:blue",
    depthshade=True,
)
ax_gt.set_title("Ground Truth")
_setup_axis(ax_gt)
try:
    fig_gt.canvas.manager.set_window_title("AtlasNet — Ground Truth")
except Exception:
    pass

fig_rec = plt.figure(figsize=(6, 6))
ax_rec = fig_rec.add_subplot(111, projection="3d")
ax_rec.scatter(
    reconstruction[:, 0],
    reconstruction[:, 1],
    reconstruction[:, 2],
    s=4,
    c="tab:orange",
    depthshade=True,
)
ax_rec.set_title("Reconstruction")
_setup_axis(ax_rec)
try:
    fig_rec.canvas.manager.set_window_title("AtlasNet — Reconstruction")
except Exception:
    pass

fig_mesh = plt.figure(figsize=(6, 6))
ax_mesh = fig_mesh.add_subplot(111, projection="3d")
tri_vertices = reconstruction[faces]
mesh = Poly3DCollection(
    tri_vertices,
    alpha=1.0,
    edgecolor="none",
    linewidths=0.0,
    facecolor=(0.78, 0.78, 0.82, 1.0),
)
mesh.set_zsort("average")
ax_mesh.add_collection3d(mesh)
ax_mesh.scatter(
    reconstruction[:, 0],
    reconstruction[:, 1],
    reconstruction[:, 2],
    s=1.5,
    c=(0.2, 0.2, 0.2, 0.2),
    depthshade=True,
)
ax_mesh.set_title("Mesh")
_setup_axis(ax_mesh)
try:
    fig_mesh.canvas.manager.set_window_title("AtlasNet — Mesh")
except Exception:
    pass

plt.show()
