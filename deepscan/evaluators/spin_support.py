"""SPIN-specific support utilities.

This file is vendored from the SPIN reference repo (`tools/activation_hooks.py`)
to ensure importance scoring matches the original implementation.

It is intentionally *not* a general framework utility; it's used by the SPIN
diagnosis evaluator (and optionally any SPIN mitigation code you may add later).
"""

from __future__ import annotations

from functools import reduce

import torch
import torch.nn as nn


class ActLinear(nn.Module):
    """Drop-in replacement of ``nn.Linear`` that records activation norms."""

    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.activation_norms = torch.zeros(
            [base.in_features], device=self.base.weight.device, requires_grad=False
        )
        self.n_samples = 0
        self.record_activation = True

    def clear_act_buffer(self):
        self.activation_norms.fill_(0.0)
        self.n_samples = 0

    def forward(self, x):
        if self.record_activation:
            if hasattr(self, "mask") and self.mask is not None:
                x_ = x[self.mask]
            else:
                x_ = x

            bs = x_.nelement() // x_.shape[-1]
            self.activation_norms = self.activation_norms * (
                self.n_samples / (self.n_samples + bs)
            ) + (x_ * x_).view(-1, x_.shape[-1]).sum(dim=0) * (
                1.0 / (self.n_samples + bs)
            )
            self.n_samples += bs

        return self.base(x)


class no_act_recording:
    """Context manager that temporarily disables activation recording."""

    def __init__(self, model):
        self.model = model

    def __enter__(self):
        for _, module in self.model.named_modules():
            if isinstance(module, ActLinear):
                module.record_activation = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for _, module in self.model.named_modules():
            if isinstance(module, ActLinear):
                module.record_activation = True


def revert_act_to_linear(model):
    """Replace ``ActLinear`` layers with their original ``nn.Linear`` base."""

    for name, module in model.named_modules():
        if isinstance(module, ActLinear):
            linear_module = module.base
            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            parent_module = (
                model
                if parent_name == ""
                else reduce(getattr, parent_name.split("."), model)
            )
            setattr(parent_module, name.split(".")[-1], linear_module)
    return model


def make_act(model, verbose: bool = False):
    """Wrap every linear submodule with ``ActLinear`` to capture gradients."""

    replace_map = {
        name: ActLinear(module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    }

    for name, module in model.named_modules():
        if verbose:
            print("current:", name)
        for target_name, replacement in replace_map.items():
            parts = target_name.split(".")
            prefix, suffix = ".".join(parts[:-1]), parts[-1]
            if prefix == "":
                continue
            if name == prefix:
                if verbose:
                    print("    modifying", suffix, "inside", name)
                setattr(module, suffix, replacement)
    return model


