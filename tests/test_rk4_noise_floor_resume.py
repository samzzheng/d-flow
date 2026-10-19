"""CPU-only checks for explicit, policy-preserving interrupted-run resume."""
from io import BytesIO
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import investigate_rk4_noise_floor as pilot


class NoiseFloorResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_untagged_checkpoint_requires_explicit_matching_legacy_policy(self):
        old = {"method":"rk4", "flow_steps":100,
               "armijo_initial_step_policy":"first_iteration_only"}
        legacy = old | {"armijo_progress_policy":"legacy_rounded_rhs"}
        strict = old | {"armijo_progress_policy":"strict_decrease_nonzero_step"}
        for current, explicit in ((strict,False),(strict,True),(legacy,False)):
            with self.subTest(current=current, explicit=explicit):
                with self.assertRaisesRegex(ValueError,"protocol mismatch"):
                    pilot.validate_resume_protocol(old,current,legacy_progress=explicit)
        identity = pilot.validate_resume_protocol(old,legacy,legacy_progress=True)
        self.assertTrue(identity["normalized_missing_legacy_tag"])
        self.assertEqual(identity["original_progress_policy"],identity["resumed_progress_policy"])
        with self.assertRaisesRegex(ValueError,"protocol mismatch"):
            pilot.validate_resume_protocol(old,legacy | {"flow_steps":99},legacy_progress=True)

    def test_tagged_policy_change_is_never_normalized(self):
        legacy = {"method":"rk4", "armijo_progress_policy":"legacy_rounded_rhs"}
        strict = legacy | {"armijo_progress_policy":"strict_decrease_nonzero_step"}
        with self.assertRaisesRegex(ValueError,"protocol mismatch"):
            pilot.validate_resume_protocol(strict,legacy,legacy_progress=True)
        with self.assertRaisesRegex(ValueError,"protocol mismatch"):
            pilot.validate_resume_protocol(legacy,strict)
        self.assertFalse(pilot.validate_resume_protocol(strict,strict)["normalized_missing_legacy_tag"])

    def test_factory_default_keeps_strict_progress(self):
        for legacy in (False,True):
            with self.subTest(legacy=legacy):
                source = torch.tensor([1.],dtype=torch.float64,requires_grad=True)
                optimizer = pilot.make_optimizer(source,lambda _:1e20,legacy_progress=legacy,max_ls=1)

                def closure():
                    source.grad = torch.ones_like(source)
                    return 1e20

                optimizer.step(closure)
                self.assertEqual(optimizer.initial_step_policy,"first_iteration_only")
                self.assertEqual(float(source.detach()),0. if legacy else 1.)
                self.assertEqual(optimizer.failed_searches,0 if legacy else 1)

    def test_full_serialized_legacy_state_continues_identically(self):
        def setup(initial):
            source = initial.detach().clone().requires_grad_(True)
            weights = torch.tensor([1.,17.],dtype=source.dtype)

            def objective(candidate):
                return float((.5*weights*candidate.square()).sum().detach())

            optimizer = pilot.make_optimizer(source,objective,legacy_progress=True,
                                             lr=1.,history=8,max_ls=8,step_cap=.1)

            def closure():
                optimizer.zero_grad()
                source.grad = weights*source.detach()
                return objective(source)

            return source,optimizer,closure

        source,uninterrupted,closure = setup(torch.tensor([2.5,-2.],dtype=torch.float64))
        for _ in range(3):
            uninterrupted.step(closure)
        self.assertTrue(uninterrupted.S)
        buffer = BytesIO()
        torch.save({"source":source.detach(),
                    "optimizer":{key:getattr(uninterrupted,key) for key in pilot.OPTIMIZER_ATTRIBUTES}},buffer)
        buffer.seek(0)
        saved = torch.load(buffer,weights_only=True)
        resumed_source,resumed,resumed_closure = setup(saved["source"])
        for key,value in saved["optimizer"].items():
            setattr(resumed,key,value)
        for _ in range(4):
            uninterrupted.step(closure)
            resumed.step(resumed_closure)
            self.assertTrue(torch.equal(source,resumed_source))
            self.assertTrue(torch.equal(uninterrupted.g,resumed.g))
            self.assertEqual(uninterrupted.f,resumed.f)
            for key in pilot.OPTIMIZER_ATTRIBUTES:
                expected,actual = getattr(uninterrupted,key),getattr(resumed,key)
                if isinstance(expected,list):
                    self.assertEqual(len(expected),len(actual))
                    self.assertTrue(all(torch.equal(a,b) for a,b in zip(expected,actual)))
                elif isinstance(expected,torch.Tensor):
                    self.assertTrue(torch.equal(expected,actual))
                else:
                    self.assertEqual(expected,actual)
        self.assertEqual(resumed.step_attempts,7)


if __name__ == "__main__":
    unittest.main()
