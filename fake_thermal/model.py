import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18


class DecoderBlock(nn.Module):
    def __init__(self, incoming, skip, outgoing):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(incoming + skip, outgoing, 3, padding=1, bias=False),
            nn.GroupNorm(8, outgoing), nn.ReLU(inplace=True),
            nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
            nn.GroupNorm(8, outgoing), nn.ReLU(inplace=True))

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.layers(torch.cat([x, skip], dim=1))


class ThermalUNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.encoder = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.encoder.fc = nn.Identity()
        self.up3 = DecoderBlock(512, 256, 128)
        self.up2 = DecoderBlock(128, 128, 64)
        self.up1 = DecoderBlock(64, 64, 32)
        self.up0 = DecoderBlock(32, 64, 24)
        self.head = nn.Sequential(nn.Conv2d(24, 16, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(16, 1, 1), nn.Sigmoid())
        self.register_buffer('mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(self, rgb):
        h, w = rgb.shape[-2:]
        x = F.pad(rgb, (0, (-w) % 32, 0, (-h) % 32), mode='replicate')
        x = (x - self.mean) / self.std
        e = self.encoder
        s0 = e.relu(e.bn1(e.conv1(x)))
        s1 = e.layer1(e.maxpool(s0))
        s2 = e.layer2(s1)
        s3 = e.layer3(s2)
        x = e.layer4(s3)
        x = self.up0(self.up1(self.up2(self.up3(x, s3), s2), s1), s0)
        x = F.interpolate(x, size=(h + (-h) % 32, w + (-w) % 32),
                          mode='bilinear', align_corners=False)
        return self.head(x)[..., :h, :w]
