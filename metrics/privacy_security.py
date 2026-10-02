"""MTA-SFL Privacy & Security Cost Simulator.

Simulates computational overhead of Secure Aggregation and accuracy 
degradation of Differential Privacy to satisfy Reviewer 1's critique.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch


class PrivacySecuritySimulator:
    """Models DP noise injection and Secure Aggregation overheads."""
    
    def __init__(self, E_crypto: float = 1.2, flops_per_byte: float = 100.0) -> None:
        """
        Args:
            E_crypto: Expansion factor for Secure Aggregation (e.g. 1.2 for secret sharing).
            flops_per_byte: Computational cost to encrypt/decrypt per byte.
        """
        self.E_crypto = E_crypto
        self.flops_per_byte = flops_per_byte

    def apply_dp_noise(self, tensor: torch.Tensor, epsilon: float, delta: float = 1e-5, sensitivity: float = 1.0) -> torch.Tensor:
        """Injects Gaussian noise for Differential Privacy.
        
        Args:
            tensor: The intermediate activations or gradients.
            epsilon: Privacy budget. If <= 0, no noise is applied.
            delta: Relaxation parameter for DP.
            sensitivity: L2 sensitivity of the query/layer.
            
        Returns:
            Noisy tensor with same shape as input.
        """
        if epsilon <= 0.0:
            return tensor
            
        # sigma = sqrt(2 * ln(1.25 / delta)) * (sensitivity / epsilon)
        sigma = math.sqrt(2.0 * math.log(1.25 / delta)) * (sensitivity / epsilon)
        noise = torch.normal(mean=0.0, std=sigma, size=tensor.size(), device=tensor.device, dtype=tensor.dtype)
        return tensor + noise
        
    def simulate_secure_agg(self, payload_bytes: int, f_n_hz: float) -> Tuple[int, float]:
        """Calculates overheads for Secure Aggregation.
        
        Args:
            payload_bytes: Original payload size in bytes.
            f_n_hz: CPU/GPU frequency of the executing node in Hz.
            
        Returns:
            Tuple of (Expanded Payload Bytes, Cryptographic Computation Latency in seconds).
        """
        expanded_bytes = int(payload_bytes * self.E_crypto)
        flops = expanded_bytes * self.flops_per_byte
        compute_latency_s = flops / max(1.0, f_n_hz)
        
        return expanded_bytes, compute_latency_s

if __name__ == "__main__":
    print("Running privacy_security.py self-tests ...")
    
    sim = PrivacySecuritySimulator(E_crypto=1.5, flops_per_byte=50.0)
    
    # Test Secure Aggregation
    original_bytes = 1_000_000 # 1 MB
    f_n = 2.0e9 # 2 GHz
    exp_bytes, latency = sim.simulate_secure_agg(original_bytes, f_n)
    assert exp_bytes == 1_500_000, f"Expected 1,500,000 bytes, got {exp_bytes}"
    assert abs(latency - (1_500_000 * 50.0 / 2.0e9)) < 1e-9
    print(f"  [OK] Secure Aggregation overheads: {exp_bytes/1e6:.2f} MB, {latency*1000:.2f} ms")
    
    # Test DP
    tensor = torch.zeros(10000, dtype=torch.float32)
    epsilon = 0.5
    noisy_tensor = sim.apply_dp_noise(tensor, epsilon, delta=1e-5, sensitivity=1.0)
    
    # Theoretical variance calculation
    # sigma = sqrt(2 * ln(1.25 / 1e-5)) * (1 / 0.5)
    # sigma = sqrt(2 * 11.736) * 2 = sqrt(23.472) * 2 = 4.844 * 2 = 9.689
    # variance = sigma^2 approx 93.88
    
    theoretical_sigma = math.sqrt(2.0 * math.log(1.25 / 1e-5)) * (1.0 / epsilon)
    theoretical_var = theoretical_sigma ** 2
    empirical_var = float(torch.var(noisy_tensor).item())
    
    print(f"  [OK] DP Theoretical Var: {theoretical_var:.2f}, Empirical Var: {empirical_var:.2f}")
    assert abs(theoretical_var - empirical_var) / theoretical_var < 0.1, "DP Variance mismatch > 10%"
    
    print("[PASS] privacy_security.py — All assertions passed.")
