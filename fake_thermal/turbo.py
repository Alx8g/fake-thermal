"""One-step RGB-to-thermal translation on SD-Turbo (pix2pix-Turbo).

Adapted from img2img-turbo (Parmar, Park, Narasimhan, Zhu, "One-Step Image
Translation with Text-to-Image Models", 2024; MIT licence,
https://github.com/GaParmar/img2img-turbo). Changes from upstream: no
hard-coded CUDA (runs on CUDA, Apple MPS or CPU), no checkpoint downloads, and
a ``translate`` helper for arbitrary frame sizes.

The base model is stabilityai/sd-turbo, downloaded from Hugging Face on first
use; check its licence before any commercial use.
"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

BASE = 'stabilityai/sd-turbo'
VAE_TARGETS = ['conv1', 'conv2', 'conv_in', 'conv_shortcut', 'conv', 'conv_out',
               'skip_conv_1', 'skip_conv_2', 'skip_conv_3', 'skip_conv_4',
               'to_k', 'to_q', 'to_v', 'to_out.0']
UNET_TARGETS = ['to_k', 'to_q', 'to_v', 'to_out.0', 'conv', 'conv1', 'conv2', 'conv_shortcut',
                'conv_out', 'proj_in', 'proj_out', 'ff.net.2', 'ff.net.0.proj']
PROMPT = 'a thermal infrared camera image, white hot'
# Each training source is a different thermal camera with its own sharpness, noise
# and contrast. Trained with one prompt per camera, the model can render one
# camera's look instead of an average of all of them; PROMPT stays the generic one.
CAMERA_PROMPTS = {'flir': PROMPT + ', FLIR ADAS vehicle camera',
                  'kaist': PROMPT + ', KAIST multispectral vehicle rig',
                  'llvip': PROMPT + ', LLVIP street surveillance camera',
                  'sf': PROMPT + ', FLIR T540 handheld camera, indoor portrait'}
SOURCE_CAMERA = {'flir': 'flir', 'video': 'flir', 'kaist': 'kaist', 'llvip': 'llvip', 'sf': 'sf'}
PEFT_DEFAULT_ALPHA = 8


def camera_prompt(camera=None):
    """Prompt for a camera name (see CAMERA_PROMPTS); None or 'none' is the generic prompt."""
    if camera in (None, '', 'none'):
        return PROMPT
    if camera not in CAMERA_PROMPTS:
        raise ValueError(f'Unknown camera {camera!r}; choose from {sorted(CAMERA_PROMPTS)} or none')
    return CAMERA_PROMPTS[camera]


def source_prompt(stem):
    """Camera prompt for a training or held-out frame id such as 'kaist_01234'."""
    return camera_prompt(SOURCE_CAMERA[stem.split('_', 1)[0]])


def grow_lora(state, rank, generator=None):
    """Raise the rank of every LoRA pair in a state dict without changing what it computes.

    lora_A (rank x in) gains rows with small random weights, lora_B (out x rank) gains
    zero columns, so B @ A is unchanged and the new rank can still learn. The caller
    must keep the LoRA scale (alpha / rank) the same, i.e. scale alpha with the rank."""
    grown = {}
    for key, value in state.items():
        if '.lora_A.' in key and value.shape[0] < rank:
            extra = torch.randn((rank - value.shape[0],) + tuple(value.shape[1:]), generator=generator,
                                dtype=torch.float32).to(value.dtype) / rank
            value = torch.cat([value, extra], 0)
        elif '.lora_B.' in key and value.shape[1] < rank:
            extra = torch.zeros((value.shape[0], rank - value.shape[1]) + tuple(value.shape[2:]), dtype=value.dtype)
            value = torch.cat([value, extra], 1)
        grown[key] = value
    return grown


def _run(module, block, *inputs):
    """Activation checkpointing for VAE blocks while training: the 512px decoder
    activations alone overflow a 24 GB GPU at batch 2."""
    if getattr(module, 'checkpointing', False) and module.training and torch.is_grad_enabled():
        return checkpoint(block, *inputs, use_reentrant=False)
    return block(*inputs)


def _encoder_forward(self, sample):
    sample = self.conv_in(sample)
    blocks = []
    for down_block in self.down_blocks:
        blocks.append(sample)
        sample = _run(self, down_block, sample)
    sample = self.mid_block(sample)
    sample = self.conv_act(self.conv_norm_out(sample))
    sample = self.conv_out(sample)
    self.current_down_blocks = blocks
    return sample


def _decoder_forward(self, sample, latent_embeds=None):
    sample = self.conv_in(sample)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    sample = self.mid_block(sample, latent_embeds).to(upscale_dtype)
    skips = [self.skip_conv_1, self.skip_conv_2, self.skip_conv_3, self.skip_conv_4]
    for index, up_block in enumerate(self.up_blocks):
        if not self.ignore_skip:
            sample = sample + skips[index](self.incoming_skip_acts[::-1][index] * self.gamma)
        sample = _run(self, up_block, sample, latent_embeds)
    sample = self.conv_norm_out(sample) if latent_embeds is None else self.conv_norm_out(sample, latent_embeds)
    return self.conv_out(self.conv_act(sample))


class Pix2PixTurbo(nn.Module):
    """SD-Turbo UNet + VAE with LoRA adapters and encoder-to-decoder skip
    connections; a single denoising step maps the input latent to the output."""

    def __init__(self, device='cuda', weights=None, lora_rank_unet=None, lora_rank_vae=None):
        """``lora_rank_*`` set the LoRA ranks of a new model (default 8 and 4). With
        ``weights``, a rank above the saved one grows the saved adapters (see
        ``grow_lora``) so training can continue with more capacity."""
        super().__init__()
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        from peft import LoraConfig
        from transformers import AutoTokenizer, CLIPTextModel
        self.device_name = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(BASE, subfolder='tokenizer')
        self.text_encoder = CLIPTextModel.from_pretrained(BASE, subfolder='text_encoder').requires_grad_(False)
        self.sched = DDPMScheduler.from_pretrained(BASE, subfolder='scheduler')
        self.sched.set_timesteps(1, device=self.device_name)
        self.sched.alphas_cumprod = self.sched.alphas_cumprod.to(self.device_name)
        vae = AutoencoderKL.from_pretrained(BASE, subfolder='vae')
        vae.encoder.forward = _encoder_forward.__get__(vae.encoder, vae.encoder.__class__)
        vae.decoder.forward = _decoder_forward.__get__(vae.decoder, vae.decoder.__class__)
        for name, (cin, cout) in {'skip_conv_1': (512, 512), 'skip_conv_2': (256, 512),
                                  'skip_conv_3': (128, 512), 'skip_conv_4': (128, 256)}.items():
            setattr(vae.decoder, name, nn.Conv2d(cin, cout, 1, bias=False))
        vae.decoder.ignore_skip = False
        vae.decoder.gamma = 1
        unet = UNet2DConditionModel.from_pretrained(BASE, subfolder='unet')
        if weights is None:
            for name in ('skip_conv_1', 'skip_conv_2', 'skip_conv_3', 'skip_conv_4'):
                nn.init.constant_(getattr(vae.decoder, name).weight, 1e-5)
            rank_unet, rank_vae = lora_rank_unet or 8, lora_rank_vae or 4
            alpha_unet = alpha_vae = PEFT_DEFAULT_ALPHA
            unet_targets, vae_targets = UNET_TARGETS, VAE_TARGETS
        else:
            state = torch.load(weights, map_location='cpu', weights_only=False)
            rank_unet, rank_vae = state['rank_unet'], state['rank_vae']
            alpha_unet = state.get('alpha_unet', PEFT_DEFAULT_ALPHA)
            alpha_vae = state.get('alpha_vae', PEFT_DEFAULT_ALPHA)
            unet_targets, vae_targets = state['unet_lora_target_modules'], state['vae_lora_target_modules']
            if lora_rank_unet and lora_rank_unet > rank_unet:
                state['state_dict_unet'] = grow_lora(state['state_dict_unet'], lora_rank_unet)
                alpha_unet, rank_unet = alpha_unet * lora_rank_unet / rank_unet, lora_rank_unet
            if lora_rank_vae and lora_rank_vae > rank_vae:
                state['state_dict_vae'] = grow_lora(state['state_dict_vae'], lora_rank_vae)
                alpha_vae, rank_vae = alpha_vae * lora_rank_vae / rank_vae, lora_rank_vae
        vae.add_adapter(LoraConfig(r=rank_vae, lora_alpha=alpha_vae, init_lora_weights='gaussian',
                                   target_modules=vae_targets), adapter_name='vae_skip')
        unet.add_adapter(LoraConfig(r=rank_unet, lora_alpha=alpha_unet, init_lora_weights='gaussian',
                                    target_modules=unet_targets))
        if weights is not None:
            for module, key in ((vae, 'state_dict_vae'), (unet, 'state_dict_unet')):
                merged = module.state_dict()
                merged.update(state[key])
                module.load_state_dict(merged)
        self.lora_rank_unet, self.lora_rank_vae = rank_unet, rank_vae
        self.lora_alpha_unet, self.lora_alpha_vae = alpha_unet, alpha_vae
        self.target_modules_unet, self.target_modules_vae = unet_targets, vae_targets
        self.unet, self.vae = unet, vae
        self.timesteps = torch.tensor([999], device=self.device_name).long()
        self.to(self.device_name)
        self._prompt_cache = {}

    def half(self):
        """Store UNet and VAE weights in fp16 for inference. On Apple MPS this is
        about 13% faster than fp16 autocast over fp32 weights, with a mean output
        difference near 5e-4."""
        self.unet.half()
        self.vae.half()
        return self

    def enable_checkpointing(self):
        self.unet.enable_gradient_checkpointing()
        self.vae.encoder.checkpointing = self.vae.decoder.checkpointing = True

    def trainable_parameters(self):
        params = [p for n, p in self.unet.named_parameters() if 'lora' in n]
        params += list(self.unet.conv_in.parameters())
        params += [p for n, p in self.vae.named_parameters() if 'lora' in n and 'vae_skip' in n]
        for name in ('skip_conv_1', 'skip_conv_2', 'skip_conv_3', 'skip_conv_4'):
            params += list(getattr(self.vae.decoder, name).parameters())
        return params

    def set_train(self):
        self.unet.train()
        self.vae.train()
        self.requires_grad_(False)
        for p in self.trainable_parameters():
            p.requires_grad_(True)

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.requires_grad_(False)

    def encode_prompt(self, prompt, batch):
        if prompt not in self._prompt_cache:
            tokens = self.tokenizer(prompt, max_length=self.tokenizer.model_max_length, padding='max_length',
                                    truncation=True, return_tensors='pt').input_ids.to(self.device_name)
            with torch.no_grad():
                self._prompt_cache[prompt] = self.text_encoder(tokens)[0]
        return self._prompt_cache[prompt].expand(batch, -1, -1)

    def forward(self, rgb, prompt=PROMPT):
        """rgb in [0, 1], Bx3xHxW with H, W multiples of 64; returns [-1, 1].
        ``prompt`` is one string for the batch or one per item."""
        if isinstance(prompt, str):
            caption = self.encode_prompt(prompt, rgb.shape[0])
        else:
            caption = torch.cat([self.encode_prompt(p, 1) for p in prompt])
        caption = caption.to(self.unet.dtype)
        latent = self.vae.encode(rgb * 2 - 1).latent_dist.mode() * self.vae.config.scaling_factor
        prediction = self.unet(latent, self.timesteps, encoder_hidden_states=caption).sample
        denoised = self.sched.step(prediction, self.timesteps, latent, return_dict=True).prev_sample
        self.vae.decoder.incoming_skip_acts = self.vae.encoder.current_down_blocks
        return self.vae.decode(denoised.to(prediction.dtype) / self.vae.config.scaling_factor).sample.clamp(-1, 1)

    def save_weights(self, path):
        state = {'unet_lora_target_modules': self.target_modules_unet,
                 'vae_lora_target_modules': self.target_modules_vae,
                 'rank_unet': self.lora_rank_unet, 'rank_vae': self.lora_rank_vae,
                 'alpha_unet': self.lora_alpha_unet, 'alpha_vae': self.lora_alpha_vae,
                 'state_dict_unet': {k: v.detach().cpu() for k, v in self.unet.state_dict().items()
                                     if 'lora' in k or 'conv_in' in k},
                 'state_dict_vae': {k: v.detach().cpu() for k, v in self.vae.state_dict().items()
                                    if 'lora' in k or 'skip' in k}}
        temporary = path.with_suffix('.tmp')
        torch.save(state, temporary)
        temporary.replace(path)

    @torch.no_grad()
    def translate(self, rgb, dtype=None, skip=1., prompt=PROMPT):
        """HxWx3 uint8 RGB to an HxW thermal map in [0, 1] (any frame size).
        ``skip`` scales the encoder-to-decoder skip activations: below 1 it
        passes less fine RGB structure (paint, glare, glass) into the output.
        ``prompt`` picks a camera look on models trained with camera prompts."""
        import numpy as np
        from torch.nn import functional as F
        h, w = rgb.shape[:2]
        x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None].float().div(255)
        weights = self.vae.dtype
        x = x.to(self.device_name, weights)
        x = F.pad(x, (0, (-w) % 64, 0, (-h) % 64), mode='replicate')
        self.vae.decoder.gamma = skip
        autocast = dtype is not None and weights == torch.float32
        try:
            with torch.autocast(self.device_name.type, dtype=dtype or torch.float16, enabled=autocast):
                y = self.forward(x, prompt).float()
        finally:
            self.vae.decoder.gamma = 1
        return ((y[0].mean(0)[:h, :w] + 1) / 2).clamp(0, 1).cpu().numpy()
