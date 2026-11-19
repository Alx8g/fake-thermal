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


class StyleConditioner(nn.Module):
    """Per-image luminance statistics to FiLM scale/shift for each encoder
    stage; the head is initialized so modulation starts as the identity."""

    def __init__(self, channels=(64, 64, 128, 256, 512)):
        super().__init__()
        self.channels = channels
        self.head = nn.Sequential(nn.Linear(4, 32), nn.ReLU(inplace=True),
                                  nn.Linear(32, 2 * sum(channels)))
        nn.init.zeros_(self.head[-1].weight)
        with torch.no_grad():  # identity FiLM at init: scale=1, shift=0
            self.head[-1].bias.copy_(torch.cat(
                [torch.cat([torch.ones(c), torch.zeros(c)]) for c in channels]))

    def stats(self, rgb):
        gray = rgb.mean(dim=1).flatten(1)
        return torch.stack([gray.mean(dim=1), gray.std(dim=1),
                            torch.quantile(gray, .05, dim=1),
                            torch.quantile(gray, .95, dim=1)], dim=1)

    def forward(self, rgb):
        style = self.head(self.stats(rgb))
        films, offset = [], 0
        for c in self.channels:
            films.append((style[:, offset:offset + c], style[:, offset + c:offset + 2 * c]))
            offset += 2 * c
        return films


FULLRES_CHANNELS = 16


class ThermalUNet(nn.Module):
    """ResNet18 U-Net. ``fullres`` adds a small full-resolution RGB feature path
    into the head; without it the finest features the head sees are a bilinear
    upsample of stride-2 decoder output, which caps pixel-level detail."""

    def __init__(self, pretrained=True, conditioning=False, fullres=False):
        super().__init__()
        self.conditioning = conditioning
        self.encoder = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.encoder.fc = nn.Identity()
        self.up3 = DecoderBlock(512, 256, 128)
        self.up2 = DecoderBlock(128, 128, 64)
        self.up1 = DecoderBlock(64, 64, 32)
        self.up0 = DecoderBlock(32, 64, 24)
        self.fullres = nn.Sequential(
            nn.Conv2d(3, FULLRES_CHANNELS, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(FULLRES_CHANNELS, FULLRES_CHANNELS, 3, padding=1),
            nn.ReLU(inplace=True)) if fullres else None
        head_in = 24 + (FULLRES_CHANNELS if fullres else 0)
        self.head = nn.Sequential(nn.Conv2d(head_in, 16, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(16, 1, 1), nn.Sigmoid())
        self.style = StyleConditioner() if conditioning else None
        self.register_buffer('mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    @staticmethod
    def options_for(state):
        """Architecture flags implied by a saved state dict."""
        return {'conditioning': any(key.startswith('style.') for key in state),
                'fullres': any(key.startswith('fullres.') for key in state)}

    @classmethod
    def from_state(cls, state, **overrides):
        """Build the architecture a state dict was saved from and load it."""
        model = cls(pretrained=False, **{**cls.options_for(state), **overrides})
        model.load_compatible(state)
        return model

    def load_compatible(self, state):
        """Strict load, except that a checkpoint without the full-resolution
        path can initialize a model that has one: the new head inputs start at
        zero, so the warm-started model reproduces the old output exactly."""
        own = self.state_dict()
        state = dict(state)
        key = 'head.0.weight'
        if key in state and state[key].shape != own[key].shape:
            old = state[key]
            if old.shape[1] > own[key].shape[1] or old.shape[0] != own[key].shape[0]:
                raise ValueError(f'Incompatible {key}: {tuple(old.shape)} vs {tuple(own[key].shape)}')
            grown = torch.zeros_like(own[key])
            grown[:, :old.shape[1]] = old
            state[key] = grown
            for name, value in own.items():
                if name.startswith('fullres.') and name not in state:
                    state[name] = value
        self.load_state_dict(state, strict=True)
        return self

    def film(self, x, params):
        scale, shift = params
        return x if scale is None else x * scale[:, :, None, None] + shift[:, :, None, None]

    def forward(self, rgb):
        h, w = rgb.shape[-2:]
        films = self.style(rgb) if self.conditioning else [(None, None)] * 5
        x = F.pad(rgb, (0, (-w) % 32, 0, (-h) % 32), mode='replicate')
        x = (x - self.mean) / self.std
        detail = self.fullres(x) if self.fullres is not None else None
        e = self.encoder
        s0 = self.film(e.relu(e.bn1(e.conv1(x))), films[0])
        s1 = self.film(e.layer1(e.maxpool(s0)), films[1])
        s2 = self.film(e.layer2(s1), films[2])
        s3 = self.film(e.layer3(s2), films[3])
        x = self.film(e.layer4(s3), films[4])
        x = self.up0(self.up1(self.up2(self.up3(x, s3), s2), s1), s0)
        x = F.interpolate(x, size=(h + (-h) % 32, w + (-w) % 32),
                          mode='bilinear', align_corners=False)
        if detail is not None:
            x = torch.cat([x, detail], dim=1)
        return self.head(x)[..., :h, :w]
