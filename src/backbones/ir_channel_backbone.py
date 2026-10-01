import torch
from .depth_channel_backbone import DepthColorBackbone


class IRBackbone(DepthColorBackbone):
    """Same frame plus temporal design as Depth_Color, single input channel.
    Kept as a separate class so its width, depth and token count can be tuned
    independently without touching the depth branch."""

    def __init__(self, in_channels=1, widths=(32, 64, 128, 256),
                 embed_dim=256, num_tokens=4):
        super().__init__(in_channels=in_channels, widths=widths,
                         embed_dim=embed_dim, num_tokens=num_tokens)


def build(**kwargs):
    return IRBackbone(**kwargs)


if __name__ == "__main__":
    m = build()
    y = m(torch.randn(2, 16, 1, 112, 112))
    print("tokens", tuple(y.shape))
