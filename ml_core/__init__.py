"""MTA-SFL ML Core Module.

Provides the distributed deep learning engine:
  - Dataset partitioning (IID, Dirichlet Non-IID, Sharded)
  - Dynamic multi-tier split architectures (CNN/1D-CNN)
  - Top-k sparsification with autograd-integrated Residual Error Feedback
  - Synchronous Split Federated Learning orchestrator with FedAvg
"""

from . import data_partitioner, split_models, compression, orchestrator

__all__ = ["data_partitioner", "split_models", "compression", "orchestrator"]
