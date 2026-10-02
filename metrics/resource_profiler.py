"""MTA-SFL Onboard Resource Profiler.

Provides latency decorators and memory tracking for RL agents to satisfy
Reviewer 1's request for quantitative analysis of onboard agent resource overhead
and exact control message volume.
"""

from __future__ import annotations

import sys
import time
from functools import wraps
from typing import Any, Callable, Dict, List

import numpy as np
import torch


class AgentResourceProfiler:
    """Tracks DDQN/LinUCB onboard computation time, memory footprint, and control volume."""
    
    def __init__(self) -> None:
        self.latencies_ns: Dict[str, List[int]] = {}
        self.memory_history_bytes: Dict[str, List[int]] = {}
        self.control_overhead_bytes: List[int] = []
        
    def log_latency(self, name: str, latency_ns: int) -> None:
        """Log execution latency in nanoseconds."""
        if name not in self.latencies_ns:
            self.latencies_ns[name] = []
        self.latencies_ns[name].append(latency_ns)
        
    def estimate_memory(self, obj: Any, seen: set = None) -> int:
        """Estimate deep memory footprint of an object including PyTorch tensors."""
        if seen is None:
            seen = set()
        
        obj_id = id(obj)
        if obj_id in seen:
            return 0
        seen.add(obj_id)
        
        if isinstance(obj, torch.Tensor):
            return obj.element_size() * obj.nelement()
        elif isinstance(obj, np.ndarray):
            return obj.nbytes
        elif isinstance(obj, (list, tuple, set, frozenset)):
            return sys.getsizeof(obj) + sum(self.estimate_memory(i, seen) for i in obj)
        elif isinstance(obj, dict):
            return sys.getsizeof(obj) + sum(self.estimate_memory(k, seen) + self.estimate_memory(v, seen) for k, v in obj.items())
        elif hasattr(obj, '__dict__'):
            return sys.getsizeof(obj) + self.estimate_memory(obj.__dict__, seen)
        else:
            return sys.getsizeof(obj)

    def log_memory(self, name: str, obj: Any) -> None:
        """Log the current memory footprint of an agent component."""
        mem = self.estimate_memory(obj)
        if name not in self.memory_history_bytes:
            self.memory_history_bytes[name] = []
        self.memory_history_bytes[name].append(mem)
        
    def log_control_overhead(self, num_neighbors: int, num_features: int = 5) -> None:
        """Calculate and log the byte-size of periodic 'Hello' beaconing.
        
        Beacon Size = sizeof(float32) * num_features * num_neighbors.
        Default features (DDQN): CC, E_res, SINR, L_life, Q_load (5 features).
        """
        bytes_per_float32 = 4
        bytes_sent = bytes_per_float32 * num_features * num_neighbors
        self.control_overhead_bytes.append(bytes_sent)

# Global singleton for decorator access
_GLOBAL_PROFILER = AgentResourceProfiler()

def get_global_profiler() -> AgentResourceProfiler:
    return _GLOBAL_PROFILER

def profile_agent_latency(name: str) -> Callable:
    """Decorator to measure precise execution latency of agent methods."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter_ns()
            result = func(*args, **kwargs)
            end = time.perf_counter_ns()
            _GLOBAL_PROFILER.log_latency(name, end - start)
            return result
        return wrapper
    return decorator

if __name__ == "__main__":
    print("Running resource_profiler.py self-tests ...")
    
    @profile_agent_latency("dummy_sleep")
    def dummy_func():
        time.sleep(0.01)
        return True
        
    # Test Latency
    dummy_func()
    lat = _GLOBAL_PROFILER.latencies_ns.get("dummy_sleep", [0])[0]
    assert lat >= 10_000_000, f"Latency {lat}ns should be >= 10ms"
    print(f"  [OK] Latency decorator works: {lat / 1e6:.2f} ms")
    
    # Test Memory
    test_tensor = torch.zeros(1000, dtype=torch.float32)  # 4000 bytes
    _GLOBAL_PROFILER.log_memory("test_tensor", test_tensor)
    mem = _GLOBAL_PROFILER.memory_history_bytes["test_tensor"][0]
    assert mem >= 4000, f"Memory {mem} should be >= 4000 bytes"
    print(f"  [OK] Memory profiling works: {mem} bytes")
    
    # Test Control Volume
    _GLOBAL_PROFILER.log_control_overhead(num_neighbors=10) # 5 features * 10 neighbors * 4 bytes = 200 bytes
    ovh = _GLOBAL_PROFILER.control_overhead_bytes[0]
    assert ovh == 200, f"Overhead {ovh} != 200 bytes"
    print(f"  [OK] Control messaging overhead calculation works: {ovh} bytes")
    
    print("[PASS] resource_profiler.py — All assertions passed.")
