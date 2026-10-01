import torch
import torch.nn as nn


class ConvBlock2d(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class ConvBlock1d(nn.Module):
    def __init__(self, cin, cout, k=3, stride=1, dilation=1):
        super().__init__()
        pad = dilation * (k - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, k, stride, pad, dilation=dilation, bias=False),
            nn.BatchNorm1d(cout),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class DepthColorBackbone(nn.Module):
    """Input [B, T, 3, H, W]. A shared 2D CNN encodes each frame, temporal 1D
    convs mix frame features across time, adaptive pooling reduces the temporal
    feature map to num_tokens embeddings. Output [B, num_tokens, embed_dim]."""

    def __init__(self, in_channels=3, widths=(32, 64, 128, 256),
                 embed_dim=256, num_tokens=4):
        super().__init__()
        blocks, c = [], in_channels
        for w in widths:
            blocks += [ConvBlock2d(c, w, stride=2), ConvBlock2d(w, w)]
            c = w
        self.frame_net = nn.Sequential(*blocks)
        self.temporal = nn.Sequential(
            ConvBlock1d(c, embed_dim, k=3, dilation=1),
            ConvBlock1d(embed_dim, embed_dim, k=3, dilation=2))
        self.pool = nn.AdaptiveAvgPool1d(num_tokens)
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens

    def forward(self, x):
        b, t, c, h, w = x.shape
        f = self.frame_net(x.reshape(b * t, c, h, w))
        f = f.mean(dim=(2, 3)).reshape(b, t, -1).transpose(1, 2)
        f = self.temporal(f)
        return self.pool(f).transpose(1, 2)


def build(**kwargs):
    return DepthColorBackbone(**kwargs)


if __name__ == "__main__":
    m = build()
    y = m(torch.randn(2, 16, 3, 112, 112))
    print("tokens", tuple(y.shape))
