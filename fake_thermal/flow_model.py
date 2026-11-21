"""Phase 5: one-step conditional flow-matching head (docs/ROADMAP.md).

Replaces regression-to-mean with sampling: v(x_t, t | rgb) is the velocity of
the straight path x_t = (1-t) x_0 + t x_1 from noise x_0 ~ N(0, I) to the
target thermal map x_1. One-step sampling uses the average-velocity property
(MeanFlow, Geng et al., arXiv 2505.13447): a single evaluation at the noise
input with full-interval conditioning. TReFT (arXiv 2511.20307) reports the
same one-step recipe working for rectified-flow image translation; Shortcut
Models (Frans et al., ICLR 2025) are the two-time alternative. See
sample_onestep for an honest account of what is exact and what is approximate.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18

from fake_thermal.model import DecoderBlock

STAGE_CHANNELS = (64, 64, 128, 256, 512)  # s0, s1, s2, s3, bottleneck


def sinusoidal_embedding(t, dim=64):
    """Fourier features for t in [0, 1]; wavelengths 1 … 1e4 in t units."""
    t = t.float().reshape(-1)
    periods = torch.logspace(0, 4, dim // 2, device=t.device)
    angles = t[:, None] * (2 * math.pi / periods)[None, :]
    return torch.cat([angles.sin(), angles.cos()], dim=1)


class FlowUNet(nn.Module):
    """Velocity head v(x_t, t | rgb): the same ResNet18 encoder / DecoderBlock
    skeleton as ThermalUNet, with the noisy map concatenated to the RGB stem
    (a 4-channel conv1) and zero-initialized FiLM conditioning from the
    timestep injected into every encoder stage. No sigmoid: velocity is
    unbounded."""

    def __init__(self, pretrained=True):
        super().__init__()
        self.encoder = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.encoder.fc = nn.Identity()
        rgb_stem = self.encoder.conv1.weight.detach().clone()
        self.encoder.conv1 = nn.Conv2d(4, 64, 7, 2, 3, bias=False)
        with torch.no_grad():  # ImageNet filters on the RGB slices; x_t slice keeps its init
            self.encoder.conv1.weight[:, :3].copy_(rgb_stem)
        self.film_sizes = [2 * channels for channels in STAGE_CHANNELS]
        self.time_embed = nn.Sequential(nn.Linear(64, 256), nn.SiLU(),
                                        nn.Linear(256, sum(self.film_sizes)))
        nn.init.zeros_(self.time_embed[-1].weight)  # FiLM starts disabled: a warmstarted
        nn.init.zeros_(self.time_embed[-1].bias)    # encoder behaves like the teacher's
        self.up3 = DecoderBlock(512, 256, 128)
        self.up2 = DecoderBlock(128, 128, 64)
        self.up1 = DecoderBlock(64, 64, 32)
        self.up0 = DecoderBlock(32, 64, 24)
        self.head = nn.Sequential(nn.Conv2d(24, 16, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(16, 1, 1))
        self.register_buffer('mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(self, xt, t, rgb):
        h, w = rgb.shape[-2:]
        pad = (0, (-w) % 32, 0, (-h) % 32)
        rgb = F.pad(rgb, pad, mode='replicate')
        xt = F.pad(xt, pad, mode='replicate')
        x = torch.cat([(rgb - self.mean) / self.std, xt], dim=1)
        chunks = torch.split(self.time_embed(sinusoidal_embedding(t)), self.film_sizes, dim=1)

        def condition(x, stage):
            channels = STAGE_CHANNELS[stage]
            scale, shift = chunks[stage][:, :channels], chunks[stage][:, channels:]
            return x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

        e = self.encoder
        s0 = condition(e.relu(e.bn1(e.conv1(x))), 0)
        s1 = condition(e.layer1(e.maxpool(s0)), 1)
        s2 = condition(e.layer2(s1), 2)
        s3 = condition(e.layer3(s2), 3)
        x = condition(e.layer4(s3), 4)
        x = self.up0(self.up1(self.up2(self.up3(x, s3), s2), s1), s0)
        x = F.interpolate(x, size=(h + (-h) % 32, w + (-w) % 32),
                          mode='bilinear', align_corners=False)
        return self.head(x)[..., :h, :w]


def flow_matching_loss(model, rgb, target, x_0=None, t=None):
    """Straight-path conditional flow-matching loss (rectified flow).

    x_t = (1-t) x_0 + t x_1 with x_1 = target; regresses v(x_t, t | rgb)
    toward x_1 - x_0, the per-coupling straight-path velocity. x_0 defaults
    to N(0, I) noise; the shortcut-style alternative is x_0 = teacher
    prediction (Frans et al., ICLR 2025), which would run the paths
    teacher-mean -> target instead — this skeleton uses noise, matching the
    MeanFlow / TReFT setting. Pass x_0 and t to make the draw deterministic
    (unit tests)."""
    if x_0 is None:
        x_0 = torch.randn_like(target)
    if t is None:
        t = torch.rand(rgb.shape[0], device=rgb.device)
    t = t.float().reshape(-1)
    xt = (1 - t[:, None, None, None]) * x_0 + t[:, None, None, None] * target
    return F.mse_loss(model(xt, t, rgb), target - x_0)


@torch.inference_mode()
def sample_onestep(model, rgb, x_0=None):
    """Sample a thermal map with a single network evaluation.

    Parameterization chosen: noise-to-data straight paths (rectified-flow
    conditional flow matching). Training regresses v(x_t, t | rgb) toward
    x_1 - x_0 at t ~ U(0, 1); sampling takes one full-length Euler step from
    the noise input, evaluating the network at the interval endpoint:

        prediction = clamp(x_0 + v(x_0, t=1 | rgb), 0, 1)

    Why one step is valid — honestly:
    - If v modeled the *average* velocity over [0, 1], this step would be
      exact: the average velocity of a straight path is (x_1 - x_0)/1.
      MeanFlow (Geng et al., arXiv 2505.13447, Alg. 2) samples exactly this
      way — one evaluation at the noise input with full-interval conditioning.
      Their time axis runs data -> noise, so their `x = e - fn(e, r=0, t=1)`
      is our `x_1 = x_0 + v(x_0, t=1)` up to a sign and time flip.
    - What we actually train is standard conditional flow matching: the target
      x_1 - x_0 is t-invariant, so the minimizer is E[x_1 - x_0 | x_t, t, rgb],
      the *instantaneous* marginal velocity, and the MeanFlow JVP identity
      that pins down the average velocity is NOT enforced. The t=1 slice only
      approximates the full-interval average velocity (near t=1 the input
      x_t ~ x_1, so the output is calibrated against nearly the whole
      displacement), and (x_0, t=1) is off the training manifold, since t=1
      training inputs are x_1, not x_0. MeanFlow's own ablation (their Tab. 1a)
      shows plain flow matching degrades badly under one-step sampling for
      class-conditional ImageNet — that is the known gap of this skeleton.
    - Why the approximation is still expected to hold here: this is strongly
      conditioned *translation* — rgb fixes most of x_1, so the conditional
      velocity is nearly deterministic and the marginal-vs-conditional gap is
      small; TReFT (arXiv 2511.20307) reports rectified-flow translation
      models sampled in one step with quality approaching multi-step. The v1
      regression teacher (teacher_warmstart) supplies encoder weights, so
      even an imperfect velocity field predicts structured maps. If one-step
      sharpness disappoints, the upgrade path is the full MeanFlow target
      (add the JVP term to the loss); the model class and this function are
      unchanged.

    The prediction is clamped to [0, 1] because fake_thermal's palettes and
    per-image p1/p99 display normalization assume display-range maps. Pass
    x_0 for deterministic sampling (unit tests)."""
    if x_0 is None:
        x_0 = torch.randn(rgb.shape[0], 1, *rgb.shape[-2:], device=rgb.device, dtype=rgb.dtype)
    t = torch.ones(rgb.shape[0], device=rgb.device)
    return (x_0 + model(x_0, t, rgb)).clamp(0, 1)


def teacher_warmstart(model, teacher_state_dict):
    """Copy encoder weights from a trained ThermalUNet checkpoint into
    FlowUNet's encoder. Accepts a raw state_dict or a train.py checkpoint
    (its 'model' entry). FlowUNet's stem has 4 input channels (RGB + x_t),
    so the teacher's 3-channel conv1 is copied into the RGB slices
    ('partial'); the x_t slice keeps its random init. Returns a report —
    'copied' (exact matches), 'partial' (slice-copied), 'unmatched_model'
    (FlowUNet encoder keys the teacher lacks or cannot fill) and
    'unmatched_teacher' (teacher encoder keys FlowUNet lacks) — so nothing
    is dropped silently. Teacher decoder/head keys are out of scope for an
    encoder warmstart and not reported."""
    stored = teacher_state_dict
    if isinstance(stored.get('model'), dict):
        stored = stored['model']
    own = model.state_dict()
    teacher_encoder = {key: value for key, value in stored.items() if key.startswith('encoder.')}
    report = {'copied': [], 'partial': [], 'unmatched_model': [], 'unmatched_teacher': []}
    for key, value in teacher_encoder.items():
        target = own.get(key)
        if target is None:
            report['unmatched_teacher'].append(key)
        elif target.shape == value.shape:
            with torch.no_grad():
                target.copy_(value)
            report['copied'].append(key)
        elif target.ndim == value.ndim == 4 and target.shape[0] == value.shape[0] \
                and target.shape[2:] == value.shape[2:]:
            with torch.no_grad():
                target[:, :value.shape[1]].copy_(value)
            report['partial'].append(key)
        else:
            report['unmatched_model'].append(key)
    report['unmatched_model'].extend(key for key in own
                                     if key.startswith('encoder.') and key not in teacher_encoder)
    return report
