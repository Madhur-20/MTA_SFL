"""MTA-SFL Ablation Studies — Component-wise analysis.

Defines ablation flags and helper functions that modify the MTA-SFL
main execution loop to isolate individual component contributions:
  1. No Topological Bridge: Forces relay to compute >= 1 layer (p2 != p3)
  2. No Pseudo-Count: Disables intrinsic exploration bonus (r_int = 0)
  3. No Top-k Sparsification: Uses full dense tensor transmission
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np


@dataclass
class AblationFlags:
    """Feature toggle flags for component-wise ablation.
    
    All True = full MTA-SFL. Set to False to disable a component.
    """
    use_topo_bridge: bool = True
    use_pseudo_count: bool = True
    use_topk: bool = True
    
    @property
    def name(self) -> str:
        """Human-readable name for this configuration."""
        if self.use_topo_bridge and self.use_pseudo_count and self.use_topk:
            return "Full MTA-SFL"
        parts = []
        if not self.use_topo_bridge:
            parts.append("No Topo-Bridge")
        if not self.use_pseudo_count:
            parts.append("No Pseudo-Count")
        if not self.use_topk:
            parts.append("No Top-k")
        return "MTA-SFL (" + ", ".join(parts) + ")"


def apply_no_topo_bridge_mask(
    linucb_mask: np.ndarray,
    arm_list: List[Tuple[int, int, int]],
) -> np.ndarray:
    """Mask out split configs where p2 == p3 (topological bridge).
    
    Forces the relay UAV to compute at least one layer.
    """
    modified = linucb_mask.copy()
    for i, (p1, p2, p3) in enumerate(arm_list):
        if p2 == p3:
            modified[i] = False
    return modified


def apply_no_pseudo_count(r_int: float) -> float:
    """Force intrinsic exploration reward to zero."""
    return 0.0


def get_dense_activation_bits(L: int, cut_layer: int, base_bits: float = 1.0e6) -> float:
    """Estimate full 32-bit dense tensor size (bits) at a given cut layer.
    
    When Top-k is disabled, the full uncompressed activation tensor is
    transmitted. This returns a scaled estimate based on the cut position.
    
    For a model with L blocks, intermediate activations at layer `cut_layer`
    are assumed to scale with the channel width at that depth. A reasonable
    approximation: early layers produce larger spatial tensors but fewer 
    channels; later layers produce smaller spatial tensors with more channels.
    The net effect is roughly constant activation volume, so we use
    base_bits as the uncompressed payload.
    
    When Top-k is active at ratio k, only k * base_bits are sent.
    When Top-k is disabled, the full base_bits are sent (no compression).
    """
    return base_bits  # Full 32-bit dense, no compression


STANDARD_ABLATIONS: List[AblationFlags] = [
    AblationFlags(),  # Full MTA-SFL
    AblationFlags(use_topo_bridge=False),
    AblationFlags(use_pseudo_count=False),
    AblationFlags(use_topk=False),
]