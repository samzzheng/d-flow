"""Actual CUDA eviction must preserve products, owned outputs and shared primal."""
import unittest

import torch

from dlr_dflow.implementation import tangent_unet
from dlr_dflow.implementation.graphed_products import GraphedAProducts
from experiments.separated_shapes_117k.bounded_products import BoundedGraphedProducts
from models.unet.default import Unet


@unittest.skipUnless(torch.cuda.device_count() > 1, "Requires the approved GPU 1")
class BoundedProductsTests(unittest.TestCase):
    def test_eviction_replay_and_ownership(self) -> None:
        torch.set_num_threads(2)
        torch.cuda.set_device(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(421)
        model = Unet(ch=4, ch_mul=[1, 2], att_channels=[0, 0],
                     groups=2, dropout=0, mid_attn=False, time_embed_dim=32).cuda(1).eval().requires_grad_(False)
        x = torch.randn(1, 1, 16, 16, device="cuda:1")
        tangent_unet.GN_STATS_MODE = "native"
        tangent_unet.GN_ACTION_BACKEND = "native"
        tangent_unet.GN_PRIMAL_BACKEND = "native"
        options = dict(share_linearization=True, full_step_graphs=False, max_jvp_columns=5, max_vjp_columns=4)
        reference = GraphedAProducts(model, x, **options)
        bounded = BoundedGraphedProducts(model, x, cache_size=4, **options)
        owned = []
        errors = []
        for time in (.17, .71):
            for provider in (reference, bounded):
                provider.set_state(time, x+time)
            for width in (1, 3, 5, 8, 11, 2, 7, 13, 4, 1):
                values = torch.randn(256, width, device=x.device)
                for kind in ("A", "AT"):
                    actual = getattr(bounded, kind)(values)
                    expected = getattr(reference, kind)(values)
                    error = float((actual-expected).norm()/expected.norm())
                    errors.append(error)
                    self.assertLess(error, 2e-6)
                    self.assertLessEqual(len(bounded._graphs), 4)
                    owned.append((actual, actual.clone()))
                torch.testing.assert_close(bounded.A.last_primal, reference.A.last_primal, rtol=0, atol=0)
        for actual, snapshot in owned:
            torch.testing.assert_close(actual, snapshot, rtol=0, atol=0)
        self.assertGreater(len(reference._graphs), len(bounded._graphs))
        print(f"Maximum bounded/unbounded relative product difference: {max(errors):.3e}")


if __name__ == "__main__":
    unittest.main()
