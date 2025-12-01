
from typing import Dict, Tuple
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def _should_checkpoint(enabled: bool, training: bool, inputs: Tuple[torch.Tensor, ...]) -> bool:
    if not enabled or not training:
        return False
    return any(isinstance(t, torch.Tensor) and t.requires_grad for t in inputs)


def _checkpoint_if_needed(module: nn.Module, inputs: Tuple[torch.Tensor, ...], enabled: bool, training: bool):
    if _should_checkpoint(enabled, training, inputs):
        return checkpoint(module, *inputs)
    return module(*inputs)

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
    def __init__(self, in_ch, skip_ch, out_ch, use_checkpoint: bool = False):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)
        self.use_checkpoint = use_checkpoint
    def forward(self, x, skip):
        def _block(x, skip):
            x = self.up(x)
            x = torch.cat([x, skip], dim=1)
            return self.conv(x)
        if _should_checkpoint(self.use_checkpoint, self.training, (x, skip)):
            return checkpoint(_block, x, skip)
        return _block(x, skip)

class SharedUNetBackbone(nn.Module):
    def __init__(self, in_ch=3, base=32, use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
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
        s1 = _checkpoint_if_needed(self.down1, (x,), self.use_checkpoint, self.training); x = self.pool1(s1)
        s2 = _checkpoint_if_needed(self.down2, (x,), self.use_checkpoint, self.training); x = self.pool2(s2)
        s3 = _checkpoint_if_needed(self.down3, (x,), self.use_checkpoint, self.training); x = self.pool3(s3)
        s4 = _checkpoint_if_needed(self.down4, (x,), self.use_checkpoint, self.training); x = self.pool4(s4)
        b  = _checkpoint_if_needed(self.bottom, (x,), self.use_checkpoint, self.training)
        return b, (s1,s2,s3,s4)

class HeadUNet(nn.Module):
    def __init__(self, base=32, out_ch=1, use_checkpoint: bool = False):
        super().__init__()
        self.up1 = UpBlock(base*16, base*8, base*8, use_checkpoint=use_checkpoint)
        self.up2 = UpBlock(base*8,  base*4, base*4, use_checkpoint=use_checkpoint)
        self.up3 = UpBlock(base*4,  base*2, base*2, use_checkpoint=use_checkpoint)
        self.up4 = UpBlock(base*2,  base,   base, use_checkpoint=use_checkpoint)
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
    def __init__(self, tasks_to_out: Dict[str,int], base=32, use_checkpoint: bool = False):
        super().__init__()
        self.backbone = SharedUNetBackbone(in_ch=3, base=base, use_checkpoint=use_checkpoint)
        self.heads = nn.ModuleDict({ t: HeadUNet(base=base, out_ch=out, use_checkpoint=use_checkpoint) for t,out in tasks_to_out.items() })
    def forward(self, x, task: str):
        b, skips = self.backbone(x)
        y = self.heads[task](b, skips)
        return y
