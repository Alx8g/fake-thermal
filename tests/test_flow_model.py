import unittest
import torch
from torch import nn

from fake_thermal.flow_model import (FlowUNet, flow_matching_loss, sample_onestep,
                                     sinusoidal_embedding, teacher_warmstart)
from fake_thermal.model import ThermalUNet


class FlowModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_forward_shapes_and_odd_sizes(self):
        model = FlowUNet(pretrained=False).eval()
        with torch.no_grad():
            for h, w in ((64, 64), (33, 65), (1, 7)):
                v = model(torch.randn(1, 1, h, w), torch.rand(1), torch.rand(1, 3, h, w))
                self.assertEqual(v.shape, (1, 1, h, w))
                self.assertTrue(torch.isfinite(v).all())
        embedding = sinusoidal_embedding(torch.tensor([0., .5, 1.]))
        self.assertEqual(embedding.shape, (3, 64))
        self.assertGreater((embedding[0] - embedding[-1]).abs().max().item(), .5)

    def test_loss_is_finite_and_backpropagates(self):
        model = FlowUNet(pretrained=False)
        loss = flow_matching_loss(model, torch.rand(2, 3, 64, 64), torch.rand(2, 1, 64, 64))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))
        self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)

    def test_loss_is_zero_at_perfect_prediction(self):
        rgb = torch.rand(2, 3, 32, 32)
        target = torch.rand(2, 1, 32, 32)
        x_0 = torch.randn(2, 1, 32, 32)

        class Perfect(nn.Module):
            def forward(self, xt, t, rgb):
                return target - x_0

        loss = flow_matching_loss(Perfect(), rgb, target, x_0=x_0, t=torch.tensor([.3, .8]))
        self.assertAlmostEqual(loss.item(), 0., places=6)

    def test_sample_onestep_bounded_shape_and_determinism(self):
        model = FlowUNet(pretrained=False).eval()
        with torch.no_grad():
            for h, w in ((64, 64), (33, 65), (1, 7)):
                out = sample_onestep(model, torch.rand(1, 3, h, w))
                self.assertEqual(out.shape, (1, 1, h, w))
                self.assertTrue(torch.isfinite(out).all())
                self.assertTrue(((out >= 0) & (out <= 1)).all())
            rgb, noise = torch.rand(1, 3, 64, 64), torch.randn(1, 1, 64, 64)
            first, second = sample_onestep(model, rgb, x_0=noise), sample_onestep(model, rgb, x_0=noise)
            self.assertTrue(torch.equal(first, second))

    def test_teacher_warmstart_copies_encoder(self):
        torch.manual_seed(1)
        teacher = ThermalUNet(pretrained=False)
        with torch.no_grad():
            for p in teacher.encoder.parameters():
                p.normal_(0, .05)
        flow = FlowUNet(pretrained=False)
        report = teacher_warmstart(flow, {'model': teacher.state_dict()})
        self.assertGreater(len(report['copied']), 100)
        self.assertEqual(report['partial'], ['encoder.conv1.weight'])
        self.assertEqual(report['unmatched_model'], [])
        self.assertEqual(report['unmatched_teacher'], [])
        self.assertTrue(torch.equal(flow.encoder.conv1.weight[:, :3], teacher.encoder.conv1.weight))
        self.assertTrue(torch.equal(flow.encoder.layer1[0].conv1.weight, teacher.encoder.layer1[0].conv1.weight))
        self.assertTrue(torch.equal(flow.encoder.layer4[1].bn2.running_mean, teacher.encoder.layer4[1].bn2.running_mean))
        # raw state dict (no checkpoint wrapper) maps the same way
        flow2 = FlowUNet(pretrained=False)
        teacher_warmstart(flow2, teacher.state_dict())
        self.assertTrue(torch.equal(flow2.encoder.layer2[0].downsample[0].weight,
                                    teacher.encoder.layer2[0].downsample[0].weight))

    def test_optimizer_step_reduces_loss(self):
        torch.manual_seed(2)
        model = FlowUNet(pretrained=False)
        rgb = torch.rand(2, 3, 64, 64)
        target = torch.zeros(2, 1, 64, 64)
        x_0, t = torch.randn(2, 1, 64, 64), torch.full((2,), .5)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
        losses = [flow_matching_loss(model, rgb, target, x_0=x_0, t=t).item()]
        for _ in range(5):
            optimizer.zero_grad(set_to_none=True)
            loss = flow_matching_loss(model, rgb, target, x_0=x_0, t=t)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        self.assertTrue(all(torch.isfinite(torch.tensor(losses))))
        self.assertLess(losses[-1], losses[0])


if __name__ == '__main__':
    unittest.main()
