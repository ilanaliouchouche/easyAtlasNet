import torch
from torch import nn
from dataclasses import dataclass

@dataclass
class PointNetConfig:
    n_cls: int
    inp_dim: int = 3
    cls_hidden_dims: tuple[int] = (512, 256)
    mlp2_hidden_dims: tuple[int] = (64, 128, 1024)
    transf_hidden_dims1: tuple[int] = (64, 128, 1024)
    transf_hidden_dims2: tuple[int] = (512, 256)
    use_bn: bool = True


class FeatTransform(nn.Module):

    def __init__(self,
                 config: PointNetConfig,
                 inp_dim=None):
        super().__init__()
        self.inp_dim = inp_dim if inp_dim is not None else config.inp_dim
        self.use_bn = config.use_bn

        hidden_dims1 = [self.inp_dim] + list(config.transf_hidden_dims1)
        hidden_dims2 = [hidden_dims1[-1]] + list(config.transf_hidden_dims2)

        def conv1x1_block(cin, cout):
            layers = [nn.Conv1d(cin, cout, kernel_size=1, bias=not self.use_bn)]
            if self.use_bn:
                layers.append(nn.BatchNorm1d(cout))
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        def linear_block(fin, fout, with_act=True):
            layers = [nn.Linear(fin, fout, bias=not self.use_bn)]
            if self.use_bn:
                layers.append(nn.BatchNorm1d(fout))
            if with_act:
                layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        self.mlp1 = nn.Sequential(*[
            conv1x1_block(hidden_dims1[i], hidden_dims1[i + 1])
            for i in range(len(hidden_dims1) - 1)
        ])

        self.mlp2 = nn.Sequential(*[
            linear_block(hidden_dims2[i], hidden_dims2[i + 1])
            for i in range(len(hidden_dims2) - 1)
        ])

        self.to_matrix = nn.Linear(hidden_dims2[-1], self.inp_dim * self.inp_dim)

    def forward(self, x):

        x = self.mlp1(x)
        x = torch.max(x, dim=-1).values
        x = self.mlp2(x)
        x = self.to_matrix(x)
        T = x.view(-1, self.inp_dim, self.inp_dim)

        return T + torch.eye(self.inp_dim, device=T.device).unsqueeze(0)


class PointNetfeat(nn.Module):

    def __init__(self,
                 config: PointNetConfig):

        super().__init__()
        self.inp_dim = config.inp_dim
        self.latent_dim1 = config.mlp2_hidden_dims[0]
        self.latent_dim = config.mlp2_hidden_dims[-1]
        self.use_bn = config.use_bn

        self.transf1 = FeatTransform(config, inp_dim=self.inp_dim)
        self.transf2 = FeatTransform(config, inp_dim=self.latent_dim1)

        def conv1x1_block(cin, cout, with_act=True):
            layers = [nn.Conv1d(cin, cout, 1, bias=not self.use_bn)]
            if self.use_bn:
                layers.append(nn.BatchNorm1d(cout))
            if with_act:
                layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        self.mlp1 = conv1x1_block(self.inp_dim, self.latent_dim1)
        mlp2_hidden_dims = [self.latent_dim1] + list(config.mlp2_hidden_dims)

        self.mlp2 = nn.Sequential(
            *[conv1x1_block(mlp2_hidden_dims[i],
                            mlp2_hidden_dims[i + 1],
                            i != (len(mlp2_hidden_dims) - 2))
              for i in range(len(mlp2_hidden_dims) - 1)]
        )

    def forward(self, x):

        T_point = self.transf1(x)
        x = torch.einsum("bcd,bdn->bcn", T_point, x)
        x = self.mlp1(x)
        T_feat = self.transf2(x)
        x_feat = torch.einsum("bcd,bdn->bcn", T_feat, x)
        x = self.mlp2(x_feat)
        global_feature = torch.max(x, dim=-1).values

        return global_feature, T_feat, x_feat


class PointNetCls(nn.Module):

    def __init__(self,
                 config: PointNetConfig) -> None:

        super().__init__()
        self.config = config

        self.features_extractor = PointNetfeat(config)

        def linear_block(fin, fout, with_act=True):
            layers = [nn.Linear(fin, fout, bias=not (config.use_bn and with_act))]
            if with_act:
                if config.use_bn:
                    layers.append(nn.BatchNorm1d(fout))
                layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        cls_hidden_dims = [config.mlp2_hidden_dims[-1]] + list(config.cls_hidden_dims) + [config.n_cls]

        self.cls_layer = nn.Sequential(
            *[linear_block(cls_hidden_dims[i],
                           cls_hidden_dims[i + 1],
                           i != (len(cls_hidden_dims) - 2))
              for i in range(len(cls_hidden_dims) - 1)]
        )

    def forward(self,
                x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        x = x.transpose(2, 1)
        x, T_feat, _ = self.features_extractor(x)
        out = self.cls_layer(x)
        return out, T_feat


if __name__ == "__main__":
    config = PointNetConfig(n_cls=10)
    model = PointNetCls(config)
    model.eval()
    x = torch.randn(4, 1024, 3)
    with torch.inference_mode():
        out, T_feat = model(x)
    print(out.shape, T_feat.shape)
