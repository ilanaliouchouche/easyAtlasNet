import torch
from torch import nn
from dataclasses import dataclass
import math

@dataclass
class AtlasNetConfig:
    k_patches: int = 25
    random_grid: bool = False
    total_n_points: int = 2500
    output_dim: int = 3
    latent_dim: int = 1024
    use_bn: bool = False
    mlp_dims: tuple[int] = (1024, 512, 256, 128)


class AtlasnetDecoder(nn.Module):

    def __init__(self,
                 config: AtlasNetConfig
                 ) -> None:

        super().__init__()
        self.config = config

        def create_mlp():
            modules_mlp = [
                nn.Conv1d(config.latent_dim + 2, config.mlp_dims[0], 1),
                nn.BatchNorm1d(config.mlp_dims[0]) if config.use_bn else nn.Identity(),
                nn.ReLU(inplace=True)
            ]
            for i in range(len(config.mlp_dims) - 1):
                modules_mlp.extend([
                    nn.Conv1d(config.mlp_dims[i], config.mlp_dims[i + 1], 1),
                    nn.BatchNorm1d(config.mlp_dims[i + 1]) if config.use_bn else nn.Identity(),
                    nn.ReLU(inplace=True)
                ])
            modules_mlp.extend([
                nn.Conv1d(config.mlp_dims[-1], config.output_dim, 1),
                nn.Tanh()
            ])
            return modules_mlp

        self.k_mlps = nn.ModuleList(
            [nn.Sequential(*create_mlp()) for _ in range(config.k_patches)]
        )

    def _sampling_square(self,
                         batch_size: int,
                         device: str
                         ) -> torch.Tensor:

        if self.config.random_grid:
            grids = torch.rand(batch_size, 2, self.config.total_n_points, device=device)
        else:
            points_per_patch = self.config.total_n_points // self.config.k_patches
            x = torch.linspace(0, 1, int(math.sqrt(points_per_patch)))
            y = x.clone()
            X, Y = torch.meshgrid([x, y], indexing="xy") 
            grid = torch.stack([X, Y], dim=-1).reshape(-1, 2)[None, ...]
            grid = grid.repeat(1, self.config.k_patches, 1)
            grids = grid.expand(batch_size, -1, -1).permute(0, 2, 1).to(device)
        return grids

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:

        if x.size(-1) != self.config.latent_dim:
            raise ValueError(f"Expected latent_dim {self.config.latent_dim}, got {x.size(-1)}")

        uv_points = self._sampling_square(x.shape[0], x.device)
        x_k = torch.cat([
            x[..., None].expand(-1, -1, uv_points.shape[-1]),
            uv_points
        ], dim=1)
        patches = x_k.chunk(self.config.k_patches, -1)
        
        output = [mlp(patch) for mlp, patch in zip(self.k_mlps, patches)]
        output = torch.stack(output, dim=1).permute(0, 1, 3, 2)
        
        return output

if __name__ == "__main__":

    decoder = AtlasnetDecoder(AtlasNetConfig())
    decoder.eval()
    x = torch.randn(2, 1024)
    with torch.inference_mode():
        y = decoder(x)
    print(y)
