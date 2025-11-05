
from typing import Dict
import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class SharedUNetBackbone(nn.Module):
    def __init__(self, in_ch=3, base=64):
        super().__init__()
        self.down1 = ConvBlock(in_ch, base)       # 256
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = ConvBlock(base, base*2)      # 128
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = ConvBlock(base*2, base*4)    # 64
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = ConvBlock(base*4, base*8)    # 32
        self.pool4 = nn.MaxPool2d(2)
        self.bottom = ConvBlock(base*8, base*16)  # 16
    def forward(self, x):
        s1 = self.down1(x); x = self.pool1(s1)
        s2 = self.down2(x); x = self.pool2(s2)
        s3 = self.down3(x); x = self.pool3(s3)
        s4 = self.down4(x); x = self.pool4(s4)
        b  = self.bottom(x)
        return b, (s1,s2,s3,s4)

class HeadUNet(nn.Module):
    def __init__(self, base=64, out_ch=1):
        super().__init__()
        self.up1 = UpBlock(base*16, base*8, base*8)
        self.up2 = UpBlock(base*8,  base*4, base*4)
        self.up3 = UpBlock(base*4,  base*2, base*2)
        self.up4 = UpBlock(base*2,  base,   base)
        self.out = nn.Conv2d(base, out_ch, 1)
    def forward(self, b, skips):
        s1,s2,s3,s4 = skips
        x = self.up1(b, s4)
        x = self.up2(x, s3)
        x = self.up3(x, s2)
        x = self.up4(x, s1)
        return self.out(x)

class TaskonomyMTL(nn.Module):
    """
    Shared UNet encoder-decoder with per-task final conv layers.
    """
    def __init__(self, tasks_to_out: Dict[str,int], base=64):
        super().__init__()
        self.backbone = SharedUNetBackbone(in_ch=3, base=base)
        self.heads = nn.ModuleDict({ t: HeadUNet(base=base, out_ch=out) for t,out in tasks_to_out.items() })
    def forward(self, x, task: str):
        b, skips = self.backbone(x)
        y = self.heads[task](b, skips)
        return y
