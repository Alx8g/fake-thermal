from collections import OrderedDict
import importlib.util
from pathlib import Path
import random
import sys
import unittest

import torch
from torch import nn

from fake_thermal.turbo import (CAMERA_PROMPTS, PROMPT, camera_prompt, grow_lora, source_prompt)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def lora_state(rank, generator):
    """A LoRA state dict shaped like peft's: a Linear pair and a 3x3 Conv pair."""
    return {
        'lin.base_layer.weight': torch.randn(5, 6, generator=generator),
        'lin.lora_A.default.weight': torch.randn(rank, 6, generator=generator),
        'lin.lora_B.default.weight': torch.randn(5, rank, generator=generator),
        'conv.lora_A.default.weight': torch.randn(rank, 3, 3, 3, generator=generator),
        'conv.lora_B.default.weight': torch.randn(4, rank, 1, 1, generator=generator),
    }


def lora_delta(state, alpha):
    """What each LoRA pair adds to its layer: (alpha / rank) * B @ A."""
    a, b = state['lin.lora_A.default.weight'], state['lin.lora_B.default.weight']
    lin = alpha / a.shape[0] * b @ a
    a, b = state['conv.lora_A.default.weight'], state['conv.lora_B.default.weight']
    conv = alpha / a.shape[0] * torch.einsum('or,rikl->oikl', b[:, :, 0, 0], a)
    return lin, conv


class CameraPromptTests(unittest.TestCase):
    def test_generic_and_camera_prompts(self):
        self.assertEqual(camera_prompt(), PROMPT)
        self.assertEqual(camera_prompt('none'), PROMPT)
        self.assertEqual(camera_prompt('kaist'), CAMERA_PROMPTS['kaist'])
        self.assertEqual(len(set(CAMERA_PROMPTS.values())), len(CAMERA_PROMPTS))
        with self.assertRaises(ValueError):
            camera_prompt('boson')

    def test_source_prompt_maps_frames_to_their_camera(self):
        self.assertEqual(source_prompt('llvip_190367'), CAMERA_PROMPTS['llvip'])
        self.assertEqual(source_prompt('video_000123'), CAMERA_PROMPTS['flir'])  # same FLIR camera
        self.assertEqual(source_prompt('flir_07438'), CAMERA_PROMPTS['flir'])
        self.assertEqual(source_prompt('sf_001_1_2_5_86_0040'), CAMERA_PROMPTS['sf'])  # SpeakingFaces

    def test_training_prompts_with_dropout(self):
        from train_turbo import prompts_for
        stems = ['flir_00001', 'kaist_00002', 'llvip_00003']
        self.assertEqual(prompts_for(stems, False), PROMPT)
        self.assertEqual(prompts_for(stems, True, 0.), [source_prompt(s) for s in stems])
        self.assertEqual(prompts_for(stems, True, 1.), [PROMPT] * 3)
        random.seed(0)
        mixed = prompts_for(stems * 200, True, .15)
        self.assertAlmostEqual(mixed.count(PROMPT) / len(mixed), .15, delta=.04)


class GrowLoraTests(unittest.TestCase):
    def test_growing_keeps_every_lora_update_and_other_weights(self):
        generator = torch.Generator().manual_seed(0)
        state = lora_state(4, generator)
        grown = grow_lora(state, 8, generator)
        self.assertEqual(grown['lin.lora_A.default.weight'].shape, (8, 6))
        self.assertEqual(grown['lin.lora_B.default.weight'].shape, (5, 8))
        self.assertEqual(grown['conv.lora_A.default.weight'].shape, (8, 3, 3, 3))
        self.assertEqual(grown['conv.lora_B.default.weight'].shape, (4, 8, 1, 1))
        self.assertTrue(torch.equal(grown['lin.base_layer.weight'], state['lin.base_layer.weight']))
        # alpha scales with the rank (8 -> 16) so alpha / rank, and the update, stay the same
        for before, after in zip(lora_delta(state, 8.), lora_delta(grown, 16.)):
            torch.testing.assert_close(after, before)
        # the new rank can learn: its A rows are not zero
        self.assertGreater(grown['lin.lora_A.default.weight'][4:].abs().sum(), 0)

    def test_growing_fp16_and_to_same_rank(self):
        state = {k: v.half() for k, v in lora_state(8, torch.Generator().manual_seed(1)).items()}
        self.assertTrue(all(torch.equal(state[k], v) for k, v in grow_lora(state, 8).items()))
        grown = grow_lora(state, 16)
        self.assertEqual(grown['lin.lora_A.default.weight'].dtype, torch.float16)
        self.assertEqual(grown['conv.lora_B.default.weight'].dtype, torch.float16)

    @unittest.skipUnless(importlib.util.find_spec('peft'), 'needs peft')
    def test_grown_peft_adapters_compute_the_same(self):
        from peft import LoraConfig, inject_adapter_in_model
        torch.manual_seed(2)

        def model(rank, alpha):
            net = nn.Sequential(OrderedDict(lin=nn.Linear(6, 6), conv=nn.Conv2d(6, 4, 3, padding=1)))
            net.load_state_dict(base)
            return inject_adapter_in_model(LoraConfig(r=rank, lora_alpha=alpha, target_modules=['lin', 'conv'],
                                                      init_lora_weights='gaussian'), net)

        base = nn.Sequential(OrderedDict(lin=nn.Linear(6, 6), conv=nn.Conv2d(6, 4, 3, padding=1))).state_dict()
        small = model(4, 8)
        for name, p in small.named_parameters():
            if 'lora_B' in name:
                nn.init.normal_(p)  # peft starts B at zero; make the adapter do something
        lora = {k: v for k, v in small.state_dict().items() if 'lora' in k}
        big = model(8, 16)
        big.load_state_dict({**big.state_dict(), **grow_lora(lora, 8)})
        x = torch.randn(2, 5, 5, 6)
        run = lambda net: net.conv(net.lin(x).permute(0, 3, 1, 2))
        with torch.no_grad():
            torch.testing.assert_close(run(big), run(small), rtol=1e-4, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
