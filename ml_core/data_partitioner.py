"""IID and Non-IID (Dirichlet / Sharded) Dataset Partitioning.

Supports CIFAR-10, CIFAR-100, MNIST, Fashion-MNIST, and UCI HAR.
Returns per-client PyTorch DataLoaders plus a centralized test_loader
for network-wide global evaluation.

Includes ``audit_class_distribution()`` to empirically verify data
heterogeneity across clients.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

# ═══════════════════════════════════════════════════════════════════════════
# UCI HAR Synthetic Dataset  (real data loadable via same interface)
# ═══════════════════════════════════════════════════════════════════════════


class UCIHARDataset(Dataset):
    """UCI Human Activity Recognition dataset.

    Input shape : (9, 128)  — 9 inertial channels, 128 timesteps.
    Labels      : 0–5 for 6 activity classes.

    If *root* contains the preprocessed UCI HAR files (``X_train.npy``,
    ``y_train.npy``, etc.) they will be loaded.  Otherwise, a synthetic
    version with matching tensor shapes is generated for development and
    smoke-testing.

    To use real data, download from
    https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones
    and preprocess into ``(N, 9, 128)`` NumPy arrays.
    """

    NUM_CLASSES = 6
    NUM_CHANNELS = 9
    SEQ_LEN = 128

    def __init__(
        self,
        root: str = "./data/uci_har",
        train: bool = True,
        download: bool = False,
        num_synthetic: int = 5000,
        seed: int = 42,
    ):
        import os

        split = "train" if train else "test"
        x_path = os.path.join(root, f"X_{split}.npy")
        y_path = os.path.join(root, f"y_{split}.npy")

        if os.path.isfile(x_path) and os.path.isfile(y_path):
            self.data = torch.from_numpy(np.load(x_path)).float()
            self.targets = torch.from_numpy(np.load(y_path)).long()
        else:
            # Synthetic fallback
            rng = np.random.RandomState(seed + (0 if train else 1))
            n = num_synthetic if train else num_synthetic // 5
            self.data = torch.from_numpy(
                rng.randn(n, self.NUM_CHANNELS, self.SEQ_LEN).astype(np.float32)
            )
            self.targets = torch.from_numpy(
                rng.randint(0, self.NUM_CLASSES, size=n).astype(np.int64)
            )

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.data[idx], int(self.targets[idx])


# ═══════════════════════════════════════════════════════════════════════════
# Data Partitioner
# ═══════════════════════════════════════════════════════════════════════════

SUPPORTED_DATASETS = ("cifar10", "cifar100", "mnist", "fashion_mnist", "uci_har")


class DataPartitioner:
    """Partition a dataset among ``num_clients`` IoT devices.

    Parameters
    ----------
    dataset_name : One of ``SUPPORTED_DATASETS``.
    num_clients  : Number of IoT devices (N).
    mode         : ``"iid"`` | ``"dirichlet"`` | ``"sharded"``.
    alpha        : Dirichlet concentration (lower → more skew).  Used only
                   when ``mode="dirichlet"``.
    shards_per_client : Max classes per client for sharded mode.
    batch_size   : DataLoader batch size.
    data_root    : Root path for dataset downloads / caches.
    seed         : Random seed for reproducibility.

    Usage::

        dp = DataPartitioner("cifar10", num_clients=10, mode="dirichlet", alpha=0.5)
        train_loaders, test_loader = dp.partition()
        dist = dp.audit_class_distribution()   # (10, 10) array
    """

    def __init__(
        self,
        dataset_name: str,
        num_clients: int,
        mode: str = "iid",
        alpha: float = 0.5,
        shards_per_client: int = 2,
        batch_size: int = 32,
        data_root: str = "./data",
        seed: int = 42,
    ):
        if dataset_name not in SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset '{dataset_name}'. "
                f"Choose from {SUPPORTED_DATASETS}"
            )
        if mode not in ("iid", "dirichlet", "sharded"):
            raise ValueError(f"Unknown partitioning mode '{mode}'")

        self.dataset_name = dataset_name
        self.num_clients = num_clients
        self.mode = mode
        self.alpha = alpha
        self.shards_per_client = shards_per_client
        self.batch_size = batch_size
        self.data_root = data_root
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Populated after partition()
        self._client_indices: Optional[List[List[int]]] = None
        self._train_dataset: Optional[Dataset] = None
        self._test_dataset: Optional[Dataset] = None
        self._targets: Optional[np.ndarray] = None
        self._num_classes: int = 0

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def _load_datasets(self) -> Tuple[Dataset, Dataset]:
        """Load train and test datasets."""
        import torchvision
        import torchvision.transforms as T

        if self.dataset_name == "cifar10":
            transform = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465),
                                                               (0.2470, 0.2435, 0.2616))])
            train = torchvision.datasets.CIFAR10(self.data_root, train=True,
                                                  download=True, transform=transform)
            test = torchvision.datasets.CIFAR10(self.data_root, train=False,
                                                 download=True, transform=transform)
            self._num_classes = 10

        elif self.dataset_name == "cifar100":
            transform = T.Compose([T.ToTensor(), T.Normalize((0.5071, 0.4867, 0.4408),
                                                               (0.2675, 0.2565, 0.2761))])
            train = torchvision.datasets.CIFAR100(self.data_root, train=True,
                                                   download=True, transform=transform)
            test = torchvision.datasets.CIFAR100(self.data_root, train=False,
                                                  download=True, transform=transform)
            self._num_classes = 100

        elif self.dataset_name == "mnist":
            transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
            train = torchvision.datasets.MNIST(self.data_root, train=True,
                                                download=True, transform=transform)
            test = torchvision.datasets.MNIST(self.data_root, train=False,
                                               download=True, transform=transform)
            self._num_classes = 10

        elif self.dataset_name == "fashion_mnist":
            transform = T.Compose([T.ToTensor(), T.Normalize((0.2860,), (0.3530,))])
            train = torchvision.datasets.FashionMNIST(self.data_root, train=True,
                                                       download=True, transform=transform)
            test = torchvision.datasets.FashionMNIST(self.data_root, train=False,
                                                      download=True, transform=transform)
            self._num_classes = 10

        elif self.dataset_name == "uci_har":
            train = UCIHARDataset(root=f"{self.data_root}/uci_har", train=True)
            test = UCIHARDataset(root=f"{self.data_root}/uci_har", train=False)
            self._num_classes = UCIHARDataset.NUM_CLASSES

        else:
            raise ValueError(f"Unsupported: {self.dataset_name}")

        return train, test

    @staticmethod
    def _extract_targets(dataset: Dataset) -> np.ndarray:
        """Extract integer labels as a NumPy array."""
        if hasattr(dataset, "targets"):
            targets = dataset.targets
            if isinstance(targets, torch.Tensor):
                return targets.numpy()
            return np.array(targets)
        elif hasattr(dataset, "labels"):
            return np.array(dataset.labels)
        else:
            # Fallback: iterate (slow but universal)
            return np.array([int(dataset[i][1]) for i in range(len(dataset))])

    # ------------------------------------------------------------------
    # Partitioning algorithms
    # ------------------------------------------------------------------

    def _iid_split(self, targets: np.ndarray) -> List[List[int]]:
        """Uniform IID split: randomly shuffle and divide equally."""
        N = len(targets)
        indices = self.rng.permutation(N).tolist()
        shard_size = N // self.num_clients
        client_indices: List[List[int]] = []
        for i in range(self.num_clients):
            start = i * shard_size
            end = start + shard_size if i < self.num_clients - 1 else N
            client_indices.append(indices[start:end])
        return client_indices

    def _dirichlet_split(self, targets: np.ndarray) -> List[List[int]]:
        """Non-IID Dirichlet split.

        For each class c, draw proportions  q_c ~ Dir(α · 1_N).
        Allocate samples of class c to clients according to q_c.
        """
        num_classes = self._num_classes
        client_indices: List[List[int]] = [[] for _ in range(self.num_clients)]

        for c in range(num_classes):
            class_idx = np.where(targets == c)[0]
            self.rng.shuffle(class_idx)

            # Draw Dirichlet proportions
            proportions = self.rng.dirichlet(
                np.full(self.num_clients, self.alpha)
            )

            # Convert proportions to cumulative sample counts
            counts = (proportions * len(class_idx)).astype(int)
            # Distribute remainder to the client with the largest fractional part
            remainder = len(class_idx) - counts.sum()
            if remainder > 0:
                fractional = (proportions * len(class_idx)) - counts
                top_clients = np.argsort(-fractional)[:remainder]
                counts[top_clients] += 1

            offset = 0
            for client_id in range(self.num_clients):
                end = offset + counts[client_id]
                client_indices[client_id].extend(class_idx[offset:end].tolist())
                offset = end

        return client_indices

    def _sharded_split(self, targets: np.ndarray) -> List[List[int]]:
        """Pathological sharded split (class-aware).

        Each client receives samples from *exactly* ``shards_per_client``
        distinct classes (S).  Classes are shuffled and cyclically assigned
        to clients so that each class is split equally among the clients
        that receive it.
        """
        num_classes = self._num_classes
        S = self.shards_per_client

        # Group sample indices by class
        class_indices: Dict[int, List[int]] = {}
        for c in range(num_classes):
            idx = np.where(targets == c)[0].tolist()
            self.rng.shuffle(idx)
            class_indices[c] = idx

        # Assign S classes to each client (round-robin over shuffled classes)
        class_order = self.rng.permutation(num_classes).tolist()
        client_classes: List[List[int]] = [[] for _ in range(self.num_clients)]
        slot = 0
        for c in class_order:
            # Assign this class to the next client that hasn't yet filled S slots
            assigned = False
            for _ in range(self.num_clients):
                cid = slot % self.num_clients
                slot += 1
                if len(client_classes[cid]) < S:
                    client_classes[cid].append(c)
                    assigned = True
                    break
            if not assigned:
                # Overflow: give to the client with fewest classes
                cid = min(range(self.num_clients), key=lambda i: len(client_classes[i]))
                client_classes[cid].append(c)

        # For each class, split its samples among the clients assigned to it
        class_clients: Dict[int, List[int]] = {}
        for cid, classes in enumerate(client_classes):
            for c in classes:
                class_clients.setdefault(c, []).append(cid)

        client_indices: List[List[int]] = [[] for _ in range(self.num_clients)]
        for c, cids in class_clients.items():
            idx = class_indices[c]
            per_client = len(idx) // len(cids)
            for j, cid in enumerate(cids):
                start = j * per_client
                end = start + per_client if j < len(cids) - 1 else len(idx)
                client_indices[cid].extend(idx[start:end])

        return client_indices

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def partition(self) -> Tuple[List[DataLoader], DataLoader]:
        """Execute the partitioning and return DataLoaders.

        Returns
        -------
        (client_loaders, test_loader) :
            ``client_loaders[i]`` is the training DataLoader for client *i*.
            ``test_loader`` is the centralized global test set.
        """
        train_ds, test_ds = self._load_datasets()
        self._train_dataset = train_ds
        self._test_dataset = test_ds
        self._targets = self._extract_targets(train_ds)

        if self.mode == "iid":
            self._client_indices = self._iid_split(self._targets)
        elif self.mode == "dirichlet":
            self._client_indices = self._dirichlet_split(self._targets)
        elif self.mode == "sharded":
            self._client_indices = self._sharded_split(self._targets)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Build per-client DataLoaders
        client_loaders: List[DataLoader] = []
        for indices in self._client_indices:
            subset = Subset(train_ds, indices)
            loader = DataLoader(
                subset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=0,
            )
            client_loaders.append(loader)

        test_loader = DataLoader(
            test_ds,
            batch_size=self.batch_size * 4,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

        return client_loaders, test_loader

    # ------------------------------------------------------------------
    # Audit / Diagnostics
    # ------------------------------------------------------------------

    def audit_class_distribution(self) -> np.ndarray:
        """Per-client, per-class sample count matrix.

        Returns
        -------
        dist : ndarray of shape ``(num_clients, num_classes)``
            ``dist[i, c]`` = number of samples of class *c* held by client *i*.
        """
        if self._client_indices is None or self._targets is None:
            raise RuntimeError("Call partition() before audit_class_distribution()")

        dist = np.zeros((self.num_clients, self._num_classes), dtype=np.int64)
        for i, indices in enumerate(self._client_indices):
            labels = self._targets[indices]
            for c in range(self._num_classes):
                dist[i, c] = int(np.sum(labels == c))
        return dist

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def client_sample_counts(self) -> List[int]:
        """Number of training samples per client."""
        if self._client_indices is None:
            raise RuntimeError("Call partition() first")
        return [len(idx) for idx in self._client_indices]


# ═══════════════════════════════════════════════════════════════════════════
# Self-Contained Verification
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Data Partitioner — Verification")
    print("=" * 60)

    # ---- IID partition on MNIST ----
    dp_iid = DataPartitioner("mnist", num_clients=5, mode="iid", seed=42)
    loaders_iid, test_loader = dp_iid.partition()
    dist_iid = dp_iid.audit_class_distribution()
    print("\n[IID] MNIST — 5 clients")
    print(f"  Samples per client : {dp_iid.client_sample_counts}")
    print(f"  Class distribution :\n{dist_iid}")
    # IID → roughly equal class counts per client
    std_per_class = dist_iid.std(axis=0)
    assert all(s < 200 for s in std_per_class), \
        f"IID distribution too skewed: std = {std_per_class}"
    print("  ✓ IID uniformity check passed")

    # ---- Dirichlet Non-IID on MNIST ----
    dp_dir = DataPartitioner("mnist", num_clients=5, mode="dirichlet",
                              alpha=0.1, seed=42)
    loaders_dir, _ = dp_dir.partition()
    dist_dir = dp_dir.audit_class_distribution()
    print("\n[Dirichlet α=0.1] MNIST — 5 clients")
    print(f"  Samples per client : {dp_dir.client_sample_counts}")
    print(f"  Class distribution :\n{dist_dir}")
    # Dirichlet α=0.1 → highly skewed
    max_ratio = dist_dir.max(axis=0) / np.maximum(dist_dir.sum(axis=0), 1)
    assert any(r > 0.5 for r in max_ratio), \
        "Dirichlet α=0.1 should produce skewed distributions"
    print("  ✓ Dirichlet skewness check passed")

    # ---- Sharded Non-IID on MNIST ----
    dp_shard = DataPartitioner("mnist", num_clients=5, mode="sharded",
                                shards_per_client=2, seed=42)
    loaders_sh, _ = dp_shard.partition()
    dist_sh = dp_shard.audit_class_distribution()
    print("\n[Sharded S=2] MNIST — 5 clients")
    print(f"  Samples per client : {dp_shard.client_sample_counts}")
    print(f"  Class distribution :\n{dist_sh}")
    # Each client should have at most 2 non-zero classes
    nonzero_classes = (dist_sh > 0).sum(axis=1)
    assert all(c <= 2 for c in nonzero_classes), \
        f"Sharded clients should have ≤ 2 classes, got {nonzero_classes}"
    print("  ✓ Sharded class-limit check passed")

    # ---- UCI HAR (synthetic) ----
    dp_har = DataPartitioner("uci_har", num_clients=3, mode="dirichlet",
                              alpha=0.5, seed=42)
    loaders_har, test_har = dp_har.partition()
    x_sample, y_sample = next(iter(loaders_har[0]))
    assert x_sample.shape[1:] == (9, 128), f"UCI HAR shape mismatch: {x_sample.shape}"
    print(f"\n[UCI HAR] shape = {x_sample.shape}, labels = {y_sample[:5]}")
    print("  ✓ UCI HAR shape check passed")

    # ---- DataLoader iteration check ----
    batch_x, batch_y = next(iter(loaders_iid[0]))
    assert batch_x.ndim == 4, f"Expected 4D tensor, got {batch_x.ndim}D"
    print(f"\n[DataLoader] batch shape = {batch_x.shape}, labels = {batch_y[:5]}")
    print("  ✓ DataLoader iteration check passed")

    print("\n" + "=" * 60)
    print("ALL DATA PARTITIONER TESTS PASSED")
    print("=" * 60)
