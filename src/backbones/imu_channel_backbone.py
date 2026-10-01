import torch
import torch.nn as nn


class ConvBlock1d(nn.Module):

    def __init__(self, cin, cout, k=5, stride=1, dilation=1):
        super().__init__()
        pad = dilation * (k - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, k, stride, pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(cout),
            nn.Dropout(0.2),
            nn.LeakyReLU(0.15, inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class IMUBackbone(nn.Module):
    """Input [B, 30, T]: 5 devices times acc xyz plus gyro xyz, resampled to a
    common grid by the loader. Dilated convolutions grow the receptive field to
    a few seconds, which covers the short motion patterns that separate the
    activities. Output [B, num_tokens, embed_dim]."""

    def __init__(self, in_channels=30, widths=(64, 128, 128, 256),
                 dilations=(1, 2, 4, 8), strides=(2, 2, 1, 1),
                 embed_dim=256, num_tokens=3):
        super().__init__()
        layers, c = [], in_channels
        for w, d, s in zip(widths, dilations, strides):
            layers.append(ConvBlock1d(c, w, k=5, stride=s, dilation=d))
            c = w
        layers.append(ConvBlock1d(c, embed_dim, k=3))
        self.net = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(num_tokens)
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens

    def forward(self, x):
        f = self.net(x)
        return self.pool(f).transpose(1, 2)


def build(**kwargs):
    return IMUBackbone(**kwargs)


if __name__ == "__main__":
    m = build()
    y = m(torch.randn(2, 30, 400))
    print("tokens", tuple(y.shape))
