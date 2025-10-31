import torch
from torch import nn
from collections import namedtuple


def champfer_loss(output: torch.Tensor,  # B, N, 2
                  ground_truth: torch.Tensor  # B, M, 2
                  ) -> torch.Tensor:
    
    diff = output[:, :, None, :] - ground_truth[:, None, :, :]  # B, N, M, 3
    distances = (diff**2).sum(-1)  # B, N, M

    loss_1 = distances.min(-1).values.mean() # B, N -> ,
    loss_2 = distances.min(-2).values.mean() # B, M -> ,

    LossDict = namedtuple("LossDict", ["precision", "recall", "total"])

    return LossDict(precision=loss_1, recall=loss_2, total=loss_1+loss_2)

def transform_regularizer(T: torch.Tensor) -> torch.Tensor:  # B, K, K

    K = T.size(1)
    I = torch.eye(K, device=T.device).unsqueeze(0)  # (1, K, K)
    loss = torch.mean(torch.norm(torch.bmm(T, T.transpose(1, 2)) - I, dim=(1, 2)))
    return loss


def generate_mesh_faces(H: int,
                        W: int,
                        K: int,
                        device="cpu") -> torch.Tensor:

    i = torch.arange(H - 1, device=device)[:, None]
    j = torch.arange(W - 1, device=device)[None, :]
    base = i * W + j  # (H-1, W-1)

    a = base
    b = base + W
    c = base + 1
    d = base + W + 1

    faces = torch.stack([
        torch.stack([a, b, c], dim=-1),
        torch.stack([b, d, c], dim=-1)
    ], dim=0).reshape(-1, 3)  # (2*(H-1)*(W-1), 3)

    if K > 1:
        offsets = torch.arange(K, device=device)[:, None, None] * (H * W)
        faces = faces.unsqueeze(0) + offsets  # (K, F, 3)
        faces = faces.reshape(-1, 3)  # (K*F, 3)

    return faces


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    H, W, K = 5, 5, 1

    faces = generate_mesh_faces(H, W, K)
    print(f"Faces shape: {faces.shape}")  # (F, 3)
    print(f"First faces:\n{faces[:5]}")

    x = torch.linspace(0, 1, W)
    y = torch.linspace(0, 1, H)
    X, Y = torch.meshgrid(x, y, indexing="xy")
    Z = torch.zeros_like(X)
    vertices = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)  # (H*W, 3)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    triangles = vertices[faces]
    mesh = Poly3DCollection(triangles.cpu().numpy(), alpha=0.3, edgecolor='k')
    ax.add_collection3d(mesh)

    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color="r", s=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title("Test Mesh Generation (Grid Triangulation)")
    plt.show()
