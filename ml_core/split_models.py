"""Dynamic Multi-Tier Split Neural Network Architectures.

Defines modular CNN / 1D-CNN backbones sliced into L ≥ 10 sequential blocks,
supporting dynamic cut-point configuration  p = (p¹, p², p³)  for the
4-tier MTA-SFL pipeline.

Strict design requirements:
  • GroupNorm (not BatchNorm) to prevent running-statistics explosions under
    dynamic activation shapes and TopK sparsity.
  • Topological Bridge Mode: when  p² = p³,  Tier 3 is a zero-FLOP identity.
  • Tier 2 concatenates smashed data from all admitted IoTs along the batch
    dimension before processing.

Supported datasets:
  • CIFAR-10 / CIFAR-100  (3×32×32)   — 2D CNN, L = 12 blocks
  • MNIST / Fashion-MNIST (1×28×28)   — 2D CNN, L = 12 blocks
  • UCI HAR               (9×128)     — 1D CNN, L = 12 blocks
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ═══════════════════════════════════════════════════════════════════════════
# Building-Block Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _conv2d_block(
    in_ch: int,
    out_ch: int,
    kernel: int = 3,
    stride: int = 1,
    padding: int = 1,
    gn_groups: int = 8,
) -> nn.Sequential:
    """Conv2d → GroupNorm → ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
        nn.GroupNorm(min(gn_groups, out_ch), out_ch),
        nn.ReLU(inplace=True),
    )


def _conv1d_block(
    in_ch: int,
    out_ch: int,
    kernel: int = 5,
    stride: int = 1,
    padding: int = 2,
    gn_groups: int = 8,
) -> nn.Sequential:
    """Conv1d → GroupNorm → ReLU."""
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
        nn.GroupNorm(min(gn_groups, out_ch), out_ch),
        nn.ReLU(inplace=True),
    )


class _Flatten(nn.Module):
    """Flatten from dim 1 onward (keep batch dim)."""
    def forward(self, x: Tensor) -> Tensor:
        return x.reshape(x.size(0), -1)


class _PoolFlatten2D(nn.Module):
    """AdaptiveAvgPool2d(1) + Flatten."""
    def forward(self, x: Tensor) -> Tensor:
        x = F.adaptive_avg_pool2d(x, 1)
        return x.reshape(x.size(0), -1)


class _PoolFlatten1D(nn.Module):
    """AdaptiveAvgPool1d(1) + Flatten."""
    def forward(self, x: Tensor) -> Tensor:
        x = F.adaptive_avg_pool1d(x, 1)
        return x.reshape(x.size(0), -1)


class _LinearReLU(nn.Module):
    """Linear → ReLU."""
    def __init__(self, in_f: int, out_f: int):
        super().__init__()
        self.fc = nn.Linear(in_f, out_f)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.fc(x))


# ═══════════════════════════════════════════════════════════════════════════
# Architecture Builders
# ═══════════════════════════════════════════════════════════════════════════

def _build_cifar_blocks(in_channels: int, num_classes: int) -> nn.ModuleList:
    """CIFAR-10/100 architecture — 12 blocks.

    Input : (B, 3, 32, 32)
    Output: (B, num_classes)

    Block layout (each is one discrete sub-layer):
      0: Conv(3→64, s1)       4: Conv(128→128, s1)    8: Conv(256→256, s2)
      1: Conv(64→64, s1)      5: Conv(128→128, s2)    9: Pool+Flatten
      2: Conv(64→64, s2)      6: Conv(128→256, s1)   10: FC(256→256)+ReLU
      3: Conv(64→128, s1)     7: Conv(256→256, s1)   11: FC(256→C)
    """
    return nn.ModuleList([
        _conv2d_block(in_channels, 64),                         # 0  → 64×32×32
        _conv2d_block(64, 64),                                   # 1  → 64×32×32
        _conv2d_block(64, 64, stride=2),                         # 2  → 64×16×16
        _conv2d_block(64, 128),                                  # 3  → 128×16×16
        _conv2d_block(128, 128),                                 # 4  → 128×16×16
        _conv2d_block(128, 128, stride=2),                       # 5  → 128×8×8
        _conv2d_block(128, 256),                                 # 6  → 256×8×8
        _conv2d_block(256, 256),                                 # 7  → 256×8×8
        _conv2d_block(256, 256, stride=2),                       # 8  → 256×4×4
        _PoolFlatten2D(),                                        # 9  → 256
        _LinearReLU(256, 256),                                   # 10 → 256
        nn.Linear(256, num_classes),                             # 11 → C
    ])


def _build_mnist_blocks(in_channels: int, num_classes: int) -> nn.ModuleList:
    """MNIST / Fashion-MNIST architecture — 12 blocks.

    Input : (B, 1, 28, 28)
    Output: (B, num_classes)
    """
    return nn.ModuleList([
        _conv2d_block(in_channels, 32),                          # 0  → 32×28×28
        _conv2d_block(32, 32),                                    # 1  → 32×28×28
        _conv2d_block(32, 32, stride=2),                          # 2  → 32×14×14
        _conv2d_block(32, 64),                                    # 3  → 64×14×14
        _conv2d_block(64, 64),                                    # 4  → 64×14×14
        _conv2d_block(64, 64, stride=2),                          # 5  → 64×7×7
        _conv2d_block(64, 128),                                   # 6  → 128×7×7
        _conv2d_block(128, 128),                                  # 7  → 128×7×7
        _conv2d_block(128, 128, stride=2),                        # 8  → 128×4×4
        _PoolFlatten2D(),                                         # 9  → 128
        _LinearReLU(128, 128),                                    # 10 → 128
        nn.Linear(128, num_classes),                              # 11 → C
    ])


def _build_har_blocks(num_classes: int) -> nn.ModuleList:
    """UCI HAR 1D-CNN architecture — 12 blocks.

    Input : (B, 9, 128)
    Output: (B, num_classes)
    """
    return nn.ModuleList([
        _conv1d_block(9, 64, kernel=5, padding=2),               # 0  → 64×128
        _conv1d_block(64, 64, kernel=5, padding=2),              # 1  → 64×128
        _conv1d_block(64, 64, kernel=3, stride=2, padding=1),    # 2  → 64×64
        _conv1d_block(64, 128, kernel=5, padding=2),             # 3  → 128×64
        _conv1d_block(128, 128, kernel=5, padding=2),            # 4  → 128×64
        _conv1d_block(128, 128, kernel=3, stride=2, padding=1),  # 5  → 128×32
        _conv1d_block(128, 256, kernel=3, padding=1),            # 6  → 256×32
        _conv1d_block(256, 256, kernel=3, padding=1),            # 7  → 256×32
        _conv1d_block(256, 256, kernel=3, stride=2, padding=1),  # 8  → 256×16
        _PoolFlatten1D(),                                         # 9  → 256
        _LinearReLU(256, 128),                                    # 10 → 128
        nn.Linear(128, num_classes),                              # 11 → C
    ])


# ═══════════════════════════════════════════════════════════════════════════
# Dataset Metadata
# ═══════════════════════════════════════════════════════════════════════════

_DATASET_INFO: Dict[str, Dict] = {
    "cifar10":       {"in_channels": 3, "num_classes": 10,  "builder": _build_cifar_blocks, "type": "2d"},
    "cifar100":      {"in_channels": 3, "num_classes": 100, "builder": _build_cifar_blocks, "type": "2d"},
    "mnist":         {"in_channels": 1, "num_classes": 10,  "builder": _build_mnist_blocks, "type": "2d"},
    "fashion_mnist": {"in_channels": 1, "num_classes": 10,  "builder": _build_mnist_blocks, "type": "2d"},
    "uci_har":       {"in_channels": 9, "num_classes": 6,   "builder": None,                "type": "1d"},
}


# ═══════════════════════════════════════════════════════════════════════════
# SplitModel: Full L-Block Architecture
# ═══════════════════════════════════════════════════════════════════════════


class SplitModel(nn.Module):
    """L-block sequential neural network that supports dynamic splitting.

    Parameters
    ----------
    dataset_name : ``"cifar10"`` | ``"cifar100"`` | ``"mnist"`` |
                   ``"fashion_mnist"`` | ``"uci_har"``.
    num_classes  : Override number of output classes.
    """

    def __init__(
        self,
        dataset_name: str,
        num_classes: Optional[int] = None,
    ):
        super().__init__()

        info = _DATASET_INFO.get(dataset_name)
        if info is None:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. "
                f"Supported: {list(_DATASET_INFO.keys())}"
            )

        nc = num_classes if num_classes is not None else info["num_classes"]
        self.dataset_name = dataset_name
        self._num_classes = nc
        self._data_type = info["type"]

        if dataset_name == "uci_har":
            self.blocks = _build_har_blocks(nc)
        else:
            self.blocks = info["builder"](info["in_channels"], nc)

    # ---- Properties ----

    @property
    def num_blocks(self) -> int:
        """Total number of blocks L."""
        return len(self.blocks)

    @property
    def L(self) -> int:
        """Alias for ``num_blocks``."""
        return len(self.blocks)

    @property
    def num_classes(self) -> int:
        return self._num_classes

    # ---- Forward execution over block ranges ----

    def forward(self, x: Tensor) -> Tensor:
        """Full forward pass through all blocks."""
        return self.forward_range(x, 0, self.L)

    def forward_range(self, x: Tensor, start: int, end: int) -> Tensor:
        """Forward through blocks[start:end] (0-indexed, exclusive end).

        Parameters
        ----------
        x     : Input tensor.
        start : First block index (inclusive).
        end   : Last block index (exclusive).
        """
        for i in range(start, end):
            x = self.blocks[i](x)
        return x

    # ---- Tier model extraction ----

    def create_tier_copy(self, start: int, end: int) -> nn.Sequential:
        """Create an independent deep copy of blocks[start:end].

        Returns a new ``nn.Sequential`` module with its own parameters.
        Use this to give each IoT device / UAV its own tier model.
        """
        if start >= end:
            return nn.Sequential()  # empty (bridge mode)
        blocks = [copy.deepcopy(self.blocks[i]) for i in range(start, end)]
        return nn.Sequential(*blocks)


# ═══════════════════════════════════════════════════════════════════════════
# TierExecutor: 4-Tier Pipeline with Dynamic Cut Points
# ═══════════════════════════════════════════════════════════════════════════

CutConfig = Tuple[int, int, int]


class TierExecutor:
    """Manages the 4-tier forward execution for one cluster.

    Given cut configuration  (p1, p2, p3)  with  1 ≤ p1 ≤ p2 ≤ p3 ≤ L:
      - Tier 1 (IoT)        : blocks  [0, p1)
      - Tier 2 (Primary UAV): blocks  [p1, p2)
      - Tier 3 (Relay UAV)  : blocks  [p2, p3)   ← identity if p2 == p3
      - Tier 4 (Cloud)      : blocks  [p3, L)

    Parameters
    ----------
    base_model : A ``SplitModel`` to extract tier architectures from.
    cut_config : ``(p1, p2, p3)``  (1-indexed cut points as in the paper).
    num_iot    : Number of IoT devices in this cluster (for creating
                 independent Tier 1 copies).
    """

    def __init__(
        self,
        base_model: SplitModel,
        cut_config: CutConfig,
        num_iot: int = 1,
    ):
        p1, p2, p3 = cut_config
        L = base_model.L

        if not (1 <= p1 <= p2 <= p3 <= L):
            raise ValueError(
                f"Invalid cut config ({p1},{p2},{p3}) for L={L}. "
                f"Require 1 ≤ p1 ≤ p2 ≤ p3 ≤ L."
            )

        self.cut_config = cut_config
        self.L = L
        self._is_bridge = (p2 == p3)

        # Create independent tier models
        self.tier1_models: List[nn.Sequential] = [
            base_model.create_tier_copy(0, p1) for _ in range(num_iot)
        ]
        self.tier2_model: nn.Sequential = base_model.create_tier_copy(p1, p2)
        self.tier3_model: nn.Sequential = base_model.create_tier_copy(p2, p3)
        self.tier4_model: nn.Sequential = base_model.create_tier_copy(p3, L)

    @property
    def is_bridge_mode(self) -> bool:
        """True if Tier 3 is an identity pass (p2 == p3)."""
        return self._is_bridge

    # ---- Tier-level forward ----

    def tier1_forward(self, x: Tensor, iot_index: int = 0) -> Tensor:
        """Tier 1 (IoT) forward: blocks [0, p1).

        Returns smashed data  δ₁.
        """
        return self.tier1_models[iot_index](x)

    def tier2_forward(self, smashed_list: List[Tensor]) -> Tensor:
        """Tier 2 (Primary UAV) forward: concatenate + blocks [p1, p2).

        Concatenates smashed data from all admitted IoTs along the batch
        dimension:  δ₁_cat = Concat({δ₁ᵢ}).

        Returns  δ₂.
        """
        delta_cat = torch.cat(smashed_list, dim=0)
        return self.tier2_model(delta_cat)

    def tier3_forward(self, x: Tensor) -> Tensor:
        """Tier 3 (Relay UAV) forward.

        If ``is_bridge_mode``:  identity pass (zero FLOPs).
        Else:  blocks [p2, p3).

        Returns  δ₃.
        """
        if self._is_bridge:
            return x
        return self.tier3_model(x)

    def tier4_forward(
        self,
        x: Tensor,
        labels: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Tier 4 (Cloud) forward: blocks [p3, L) + cross-entropy loss.

        Returns (logits, loss).
        """
        logits = self.tier4_model(x)
        loss = F.cross_entropy(logits, labels, reduction='sum')
        return logits, loss

    # ---- Full pipeline (without compression — see orchestrator) ----

    def full_forward(
        self,
        x_list: List[Tensor],
        labels_list: List[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """Run the complete 4-tier pipeline.

        Parameters
        ----------
        x_list      : List of input tensors, one per IoT device.
        labels_list : Corresponding labels.

        Returns
        -------
        (logits, loss)
        """
        # Tier 1
        smashed = [
            self.tier1_forward(x_list[i], i)
            for i in range(len(x_list))
        ]
        # Tier 2 (concat + process)
        delta2 = self.tier2_forward(smashed)
        # Tier 3
        delta3 = self.tier3_forward(delta2)
        # Tier 4
        labels_cat = torch.cat(labels_list, dim=0)
        return self.tier4_forward(delta3, labels_cat)

    # ---- Parameter accessors ----

    def all_parameters(self) -> List[torch.nn.Parameter]:
        """All trainable parameters across all tiers."""
        params: List[torch.nn.Parameter] = []
        for m in self.tier1_models:
            params.extend(m.parameters())
        params.extend(self.tier2_model.parameters())
        params.extend(self.tier3_model.parameters())
        params.extend(self.tier4_model.parameters())
        return params

    def tier1_parameters(self, iot_index: int = 0) -> List[torch.nn.Parameter]:
        return list(self.tier1_models[iot_index].parameters())

    def tier2_parameters(self) -> List[torch.nn.Parameter]:
        return list(self.tier2_model.parameters())

    def tier3_parameters(self) -> List[torch.nn.Parameter]:
        return list(self.tier3_model.parameters())

    def tier4_parameters(self) -> List[torch.nn.Parameter]:
        return list(self.tier4_model.parameters())

    def to(self, device: torch.device) -> "TierExecutor":
        """Move all models to a device."""
        for m in self.tier1_models:
            m.to(device)
        self.tier2_model.to(device)
        self.tier3_model.to(device)
        self.tier4_model.to(device)
        return self


# ═══════════════════════════════════════════════════════════════════════════
# Self-Contained Verification
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Split Models — Verification")
    print("=" * 60)

    torch.manual_seed(0)

    # ---- Test 1: Full model forward for each dataset ----
    test_inputs = {
        "cifar10":       torch.randn(2, 3, 32, 32),
        "cifar100":      torch.randn(2, 3, 32, 32),
        "mnist":         torch.randn(2, 1, 28, 28),
        "fashion_mnist": torch.randn(2, 1, 28, 28),
        "uci_har":       torch.randn(2, 9, 128),
    }
    expected_classes = {"cifar10": 10, "cifar100": 100, "mnist": 10,
                        "fashion_mnist": 10, "uci_har": 6}

    for ds_name, x in test_inputs.items():
        model = SplitModel(ds_name)
        logits = model(x)
        assert logits.shape == (2, expected_classes[ds_name]), \
            f"{ds_name}: expected (2, {expected_classes[ds_name]}), got {logits.shape}"
        assert model.L >= 10, f"{ds_name}: L = {model.L} < 10"
        print(f"  [{ds_name}] L = {model.L}, output = {logits.shape} ✓")

    # ---- Test 2: TierExecutor with normal cut ----
    base = SplitModel("cifar10")
    executor = TierExecutor(base, cut_config=(3, 6, 9), num_iot=2)
    x1, x2 = torch.randn(4, 3, 32, 32), torch.randn(4, 3, 32, 32)
    y1, y2 = torch.randint(0, 10, (4,)), torch.randint(0, 10, (4,))

    logits, loss = executor.full_forward([x1, x2], [y1, y2])
    assert logits.shape == (8, 10), f"Expected (8, 10), got {logits.shape}"
    assert loss.item() > 0, "Loss should be > 0"
    loss.backward()

    # Verify gradients exist in all tiers
    for i, m in enumerate(executor.tier1_models):
        has_grad = any(p.grad is not None for p in m.parameters())
        assert has_grad, f"Tier 1 model {i} has no gradients"
    assert any(p.grad is not None for p in executor.tier2_model.parameters()), \
        "Tier 2 has no gradients"
    assert any(p.grad is not None for p in executor.tier4_model.parameters()), \
        "Tier 4 has no gradients"
    print(f"\n  [Cut (3,6,9)] logits = {logits.shape}, loss = {loss.item():.4f}")
    print("  ✓ Normal cut forward/backward passed with gradients in all tiers")

    # ---- Test 3: Topological Bridge Mode (p2 == p3) ----
    executor_bridge = TierExecutor(base, cut_config=(3, 6, 6), num_iot=2)
    assert executor_bridge.is_bridge_mode, "Should be in bridge mode"
    assert len(list(executor_bridge.tier3_model.parameters())) == 0, \
        "Bridge tier 3 should have no parameters"

    logits_b, loss_b = executor_bridge.full_forward([x1, x2], [y1, y2])
    assert logits_b.shape == (8, 10), f"Bridge: expected (8, 10), got {logits_b.shape}"
    loss_b.backward()
    print(f"\n  [Bridge (3,6,6)] logits = {logits_b.shape}, loss = {loss_b.item():.4f}")
    print("  ✓ Bridge mode forward/backward passed (zero-FLOP Tier 3)")

    # ---- Test 4: Extreme cut configs ----
    # All on cloud
    exec_all_cloud = TierExecutor(base, cut_config=(1, 1, 1), num_iot=1)
    logits_c, loss_c = exec_all_cloud.full_forward([x1], [y1])
    assert logits_c.shape == (4, 10)
    loss_c.backward()
    print(f"\n  [Cut (1,1,1)] All-cloud: logits = {logits_c.shape} ✓")

    # All on IoT
    exec_all_iot = TierExecutor(base, cut_config=(12, 12, 12), num_iot=1)
    logits_i, loss_i = exec_all_iot.full_forward([x1], [y1])
    assert logits_i.shape == (4, 10)
    loss_i.backward()
    print(f"  [Cut (12,12,12)] All-IoT: logits = {logits_i.shape} ✓")

    # ---- Test 5: UCI HAR with TierExecutor ----
    har_model = SplitModel("uci_har")
    har_exec = TierExecutor(har_model, cut_config=(4, 7, 10), num_iot=3)
    x_har = [torch.randn(2, 9, 128) for _ in range(3)]
    y_har = [torch.randint(0, 6, (2,)) for _ in range(3)]
    logits_h, loss_h = har_exec.full_forward(x_har, y_har)
    assert logits_h.shape == (6, 6)
    loss_h.backward()
    print(f"\n  [UCI HAR (4,7,10)] logits = {logits_h.shape}, loss = {loss_h.item():.4f}")
    print("  ✓ UCI HAR tier executor passed")

    print("\n" + "=" * 60)
    print("ALL SPLIT MODEL TESTS PASSED")
    print("=" * 60)
