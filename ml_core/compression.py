"""True Top-k Sparsification with Autograd-Integrated Residual Error Feedback.

Implements:
  • ``topk_compress`` / ``topk_decompress``: standalone NumPy-free TopK.
  • ``TopKSparsifyWithEF``: custom ``torch.autograd.Function`` that compresses
    **both** forward activations and backward gradients through the
    PyTorch computational graph — no fake dense backward.
  • Forward compression (Section III.C): TopK by magnitude, zero-pad discarded
    elements.  No forward error feedback (prevents spatial noise corruption).
  • Backward compression with Residual Error Feedback (Eqs. 16–18):
      - Eq. 16:  g_acc = ∇δ + e_n
      - Eq. 17:  ∇δ^sparse = TopK(g_acc, k)
      - Eq. 18:  e_n ← g_acc − Decomp(∇δ^sparse)
  • ``CompressionManager``: high-level buffer registry with norm tracking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


# ═══════════════════════════════════════════════════════════════════════════
# Residual Error Buffer
# ═══════════════════════════════════════════════════════════════════════════


class ResidualErrorBuffer:
    """Persistent residual error buffer for gradient compression (Eq. 18).

    Maintains a buffer  e ∈ R^D  that accumulates discarded gradient
    energy across consecutive backward passes.
    """

    def __init__(self) -> None:
        self._buffer: Optional[Tensor] = None
        self._update_count: int = 0

    @property
    def buffer(self) -> Optional[Tensor]:
        return self._buffer

    @property
    def is_initialized(self) -> bool:
        return self._buffer is not None

    @property
    def update_count(self) -> int:
        return self._update_count

    def get_or_init(self, shape: torch.Size, device: torch.device,
                    dtype: torch.dtype) -> Tensor:
        """Return the buffer, initialising to zero if needed or if shape changed."""
        if (self._buffer is None
                or self._buffer.shape != shape
                or self._buffer.device != device):
            self._buffer = torch.zeros(shape, device=device, dtype=dtype)
        return self._buffer

    def update(self, new_residual: Tensor) -> None:
        """Store the new residual (Eq. 18) and increment counter."""
        self._buffer = new_residual.detach().clone()
        self._update_count += 1

    def reset(self) -> None:
        """Zero-out the buffer and reset counter."""
        if self._buffer is not None:
            self._buffer.zero_()
        self._update_count = 0

    def norm(self) -> float:
        """L2 norm  ‖e‖₂  of the current buffer."""
        if self._buffer is None:
            return 0.0
        return float(self._buffer.norm(p=2).item())


# ═══════════════════════════════════════════════════════════════════════════
# Standalone TopK Utilities
# ═══════════════════════════════════════════════════════════════════════════


def topk_compress(
    x: Tensor,
    k_ratio: float,
) -> Tuple[Tensor, Tensor, int]:
    """Extract top-k elements by magnitude.

    Parameters
    ----------
    x       : Arbitrary-shaped tensor.
    k_ratio : Fraction of elements to keep ∈ (0, 1].

    Returns
    -------
    (values, indices, original_numel) :
        ``values``  : 1-D tensor of kept values.
        ``indices`` : 1-D tensor of kept flat indices.
        ``original_numel`` : total number of elements in *x*.
    """
    flat = x.reshape(-1)
    k = max(1, int(math.ceil(k_ratio * flat.numel())))
    k = min(k, flat.numel())

    _, top_idx = torch.topk(flat.abs(), k, sorted=False)
    return flat[top_idx], top_idx, flat.numel()


def topk_decompress(
    values: Tensor,
    indices: Tensor,
    numel: int,
    shape: torch.Size,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """Reconstruct a tensor from its TopK representation via zero-padding.

    Parameters
    ----------
    values  : 1-D kept values.
    indices : 1-D flat indices.
    numel   : Original total element count.
    shape   : Original tensor shape.
    device  : Target device.
    dtype   : Target dtype.
    """
    if device is None:
        device = values.device
    if dtype is None:
        dtype = values.dtype
    flat = torch.zeros(numel, device=device, dtype=dtype)
    flat[indices] = values
    return flat.reshape(shape)


# ═══════════════════════════════════════════════════════════════════════════
# Custom Autograd Function  (Forward + Backward Compression)
# ═══════════════════════════════════════════════════════════════════════════


class TopKSparsifyWithEF(torch.autograd.Function):
    """TopK sparsification for both forward and backward passes.

    Forward:
        • TopK on activations (smashed data), zero-pad discarded elements.
        • **No** forward error feedback  (Section III.C).

    Backward:
        • Receive upstream gradient  ∇δ.
        • Eq. 16:  g_acc = ∇δ + e_n
        • Eq. 17:  ∇δ^sparse = TopK(g_acc, k)
        • Eq. 18:  e_n ← g_acc − Decomp(∇δ^sparse)
        • Propagate  ∇δ^sparse  downstream.

    Usage::

        ef_buf = ResidualErrorBuffer()
        compressed = TopKSparsifyWithEF.apply(activations, 0.3, ef_buf)
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: Tensor,
        k_ratio: float,
        backward_ef_buffer: ResidualErrorBuffer,
    ) -> Tensor:
        # --- Forward TopK (no EF) ---
        flat = x.reshape(-1)
        k = max(1, int(math.ceil(k_ratio * flat.numel())))
        k = min(k, flat.numel())

        _, top_idx = torch.topk(flat.abs(), k, sorted=False)

        # Zero-padded decompression
        decompressed = torch.zeros_like(flat)
        decompressed[top_idx] = flat[top_idx]
        decompressed = decompressed.reshape(x.shape)

        # Save for backward
        ctx.k_ratio = k_ratio
        ctx.backward_ef_buffer = backward_ef_buffer
        ctx.input_shape = x.shape

        return decompressed

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: Tensor,
    ) -> Tuple[Optional[Tensor], None, None]:
        if not ctx.needs_input_grad[0]:
            return None, None, None

        k_ratio = ctx.k_ratio
        ef_buf: ResidualErrorBuffer = ctx.backward_ef_buffer
        shape = ctx.input_shape

        flat_grad = grad_output.reshape(-1)

        # --- Eq. 16:  g_acc = ∇δ + e_n ---
        ef = ef_buf.get_or_init(flat_grad.shape, flat_grad.device, flat_grad.dtype)
        g_acc = flat_grad + ef

        # --- Eq. 17:  ∇δ^sparse = TopK(g_acc, k) ---
        k = max(1, int(math.ceil(k_ratio * g_acc.numel())))
        k = min(k, g_acc.numel())

        _, top_idx = torch.topk(g_acc.abs(), k, sorted=False)

        sparse_grad = torch.zeros_like(g_acc)
        sparse_grad[top_idx] = g_acc[top_idx]

        # --- Eq. 18:  e_n ← g_acc − Decomp(∇δ^sparse) ---
        new_residual = g_acc - sparse_grad
        ef_buf.update(new_residual)

        return sparse_grad.reshape(shape), None, None


# ═══════════════════════════════════════════════════════════════════════════
# Compression Manager
# ═══════════════════════════════════════════════════════════════════════════


class CompressionManager:
    """High-level manager for TopK+EF compression across multiple nodes.

    Maintains named ``ResidualErrorBuffer`` instances and provides a clean
    API for compressing tensors at tier boundaries.

    Parameters
    ----------
    k_ratio : Global default sparsity ratio (fraction of elements kept).
    """

    def __init__(self, k_ratio: float = 0.5):
        if not 0.0 < k_ratio <= 1.0:
            raise ValueError(f"k_ratio must be in (0, 1], got {k_ratio}")
        self.k_ratio = k_ratio
        self._buffers: Dict[str, ResidualErrorBuffer] = {}

    def get_buffer(self, key: str) -> ResidualErrorBuffer:
        """Get or create a named residual error buffer."""
        if key not in self._buffers:
            self._buffers[key] = ResidualErrorBuffer()
        return self._buffers[key]

    def compress(
        self,
        x: Tensor,
        buffer_key: str,
        k_ratio: Optional[float] = None,
    ) -> Tensor:
        """Apply TopK+EF compression through the autograd graph.

        Parameters
        ----------
        x          : Activation tensor to compress.
        buffer_key : Identifier for the backward EF buffer.
        k_ratio    : Override sparsity ratio.

        Returns
        -------
        Decompressed (zero-padded) tensor with same shape as *x*.
        """
        ratio = k_ratio if k_ratio is not None else self.k_ratio
        ef_buf = self.get_buffer(buffer_key)
        return TopKSparsifyWithEF.apply(x, ratio, ef_buf)

    def reset_all_buffers(self) -> None:
        """Zero-out all residual error buffers."""
        for buf in self._buffers.values():
            buf.reset()

    def all_residual_norms(self) -> Dict[str, float]:
        """Return L2 norms of all residual buffers."""
        return {key: buf.norm() for key, buf in self._buffers.items()}

    def total_buffers(self) -> int:
        return len(self._buffers)

    def buffer_summary(self) -> Dict[str, Dict[str, Any]]:
        """Detailed summary of all buffers."""
        summary: Dict[str, Dict[str, Any]] = {}
        for key, buf in self._buffers.items():
            summary[key] = {
                "norm": buf.norm(),
                "update_count": buf.update_count,
                "initialized": buf.is_initialized,
                "numel": buf.buffer.numel() if buf.buffer is not None else 0,
            }
        return summary


# ═══════════════════════════════════════════════════════════════════════════
# Self-Contained Verification
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Compression Module — Verification")
    print("=" * 60)

    torch.manual_seed(42)

    # ---- Test 1: TopK compress / decompress roundtrip ----
    x = torch.randn(4, 8)
    vals, idx, numel = topk_compress(x, k_ratio=0.5)
    x_hat = topk_decompress(vals, idx, numel, x.shape)

    # Exactly k_ratio fraction should be non-zero
    nonzero_ratio = (x_hat != 0).float().mean().item()
    assert 0.45 <= nonzero_ratio <= 0.55, \
        f"Expected ~50% nonzero, got {nonzero_ratio:.2%}"
    print(f"  [TopK] nonzero ratio = {nonzero_ratio:.2%}")
    print("  ✓ TopK compress/decompress roundtrip passed")

    # ---- Test 2: Forward compression zeros out elements ----
    ef = ResidualErrorBuffer()
    inp = torch.randn(2, 16, requires_grad=True)
    out = TopKSparsifyWithEF.apply(inp, 0.25, ef)

    fwd_nonzero = (out != 0).float().mean().item()
    assert fwd_nonzero < 0.35, \
        f"Forward should keep ~25%, got {fwd_nonzero:.2%}"
    print(f"\n  [Forward] nonzero ratio = {fwd_nonzero:.2%}")
    print("  ✓ Forward compression check passed")

    # ---- Test 3: Backward gradient is sparsified ----
    loss = out.sum()
    loss.backward()

    grad = inp.grad
    assert grad is not None, "Gradient should not be None"
    bwd_nonzero = (grad != 0).float().mean().item()
    assert bwd_nonzero < 0.35, \
        f"Backward gradient should be ~25% nonzero, got {bwd_nonzero:.2%}"
    print(f"\n  [Backward] gradient nonzero ratio = {bwd_nonzero:.2%}")
    print("  ✓ Backward gradient sparsification check passed")

    # ---- Test 4: Residual error buffer is non-trivially updated ----
    assert ef.is_initialized, "EF buffer should be initialized after backward"
    assert ef.update_count == 1, f"Expected 1 update, got {ef.update_count}"
    ef_norm = ef.norm()
    assert ef_norm > 0.0, "Residual buffer should be non-zero (stores discarded energy)"
    print(f"\n  [EF Buffer] norm = {ef_norm:.4f}, updates = {ef.update_count}")
    print("  ✓ Residual error buffer check passed")

    # ---- Test 5: EF buffer accumulates across steps ----
    norms = [ef_norm]
    for step in range(3):
        inp2 = torch.randn(2, 16, requires_grad=True)
        out2 = TopKSparsifyWithEF.apply(inp2, 0.25, ef)
        loss2 = out2.sum()
        loss2.backward()
        norms.append(ef.norm())

    print(f"\n  [EF Accumulation] norms over 4 steps: "
          f"{[f'{n:.4f}' for n in norms]}")
    assert ef.update_count == 4, f"Expected 4 updates, got {ef.update_count}"
    # After multiple steps, the buffer should have changed non-trivially
    assert not all(abs(norms[i] - norms[0]) < 1e-8 for i in range(1, len(norms))), \
        "EF buffer norms should change across steps"
    print("  ✓ Multi-step EF accumulation check passed")

    # ---- Test 6: CompressionManager API ----
    mgr = CompressionManager(k_ratio=0.3)
    inp3 = torch.randn(4, 32, requires_grad=True)
    out3 = mgr.compress(inp3, buffer_key="tier1_to_tier2")
    loss3 = out3.sum()
    loss3.backward()

    summaries = mgr.buffer_summary()
    assert "tier1_to_tier2" in summaries
    assert summaries["tier1_to_tier2"]["update_count"] == 1
    print(f"\n  [Manager] buffers = {mgr.total_buffers()}, "
          f"norms = {mgr.all_residual_norms()}")
    print("  ✓ CompressionManager API check passed")

    # ---- Test 7: Reset clears all buffers ----
    mgr.reset_all_buffers()
    assert all(n == 0.0 for n in mgr.all_residual_norms().values()), \
        "All buffer norms should be 0 after reset"
    print("\n  ✓ Buffer reset check passed")

    print("\n" + "=" * 60)
    print("ALL COMPRESSION TESTS PASSED")
    print("=" * 60)
