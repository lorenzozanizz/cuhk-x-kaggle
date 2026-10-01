import torch
import torch.nn as nn
from .imu_channel_backbone import ConvBlock1d


class SkeletonBackbone(nn.Module):
    """Input [B, T, J, 3], normalized by the loader. A pointwise conv embeds
    the flattened joints of each frame, then dilated temporal convs model the
    motion. Plain CNN family, so it stays inside the CNN/RNN/Transformer rule
    without relying on the graph convolution interpretation."""

    def __init__(self, num_joints=17, stem_dim=64,
                 widths=(64, 128, 128, 256), dilations=(1, 2, 4, 8),
                 strides=(2, 2, 1, 1), embed_dim=256, num_tokens=4):
        super().__init__()
        self.stem = ConvBlock1d(num_joints * 3, stem_dim, k=1)
        layers, c = [], stem_dim
        for w, d, s in zip(widths, dilations, strides):
            layers.append(ConvBlock1d(c, w, k=5, stride=s, dilation=d))
            c = w
        layers.append(ConvBlock1d(c, embed_dim, k=3))
        self.net = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(num_tokens)
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens

    def forward(self, x):
        b, t, j, c = x.shape
        f = x.reshape(b, t, j * c).transpose(1, 2)
        f = self.net(self.stem(f))
        return self.pool(f).transpose(1, 2)


def build(**kwargs):
    return SkeletonBackbone(**kwargs)


if __name__ == "__main__":
    m = build()
    y = m(torch.randn(2, 64, 17, 3))
    print("tokens", tuple(y.shape))
