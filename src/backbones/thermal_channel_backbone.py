import torch
from .depth_channel_backbone import DepthColorBackbone


class ThermalBackbone(DepthColorBackbone):
    """Narrower than the depth branch: thermal frames are low information and
    often missing, so the branch gets a smaller parameter budget by default."""

    def __init__(self, in_channels=1, widths=(16, 32, 64, 128),
                 embed_dim=128, num_tokens=4):
        super().__init__(in_channels=in_channels, widths=widths,
                         embed_dim=embed_dim, num_tokens=num_tokens)


def build(**kwargs):
    return ThermalBackbone(**kwargs)


if __name__ == "__main__":
    m = build()
    y = m(torch.randn(2, 16, 1, 112, 112))
    print("tokens", tuple(y.shape))
