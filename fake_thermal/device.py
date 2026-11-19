"""Shared accelerator selection; model arithmetic stays in float32."""
import torch


DEVICES = ('auto', 'cpu', 'mps', 'cuda')


def select_device(name='auto'):
    if name not in DEVICES:
        raise ValueError(f'Unknown device: {name}')
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if name == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable; use --device cpu or auto')
    if name == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS requested but unavailable; use --device cpu or auto')
    return torch.device(name)


def memory_format(device):
    return torch.contiguous_format if device.type == 'mps' else torch.channels_last


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()
