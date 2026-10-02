#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MTA-SFL  —  Multi-Tier Adaptive Split Federated Learning                  ║
║  main.py  —  Algorithm 1: Joint Execution Phase                            ║
║                                                                            ║
║  This script wires every decoupled module into the chronological sequence  ║
║  of Algorithm 1:                                                           ║
║                                                                            ║
║    1. Mobility & Channel update                                            ║
║    2. Capacity-Aware OFDMA Association                                     ║
║    3. Action Masking → DDQN (relay) → LinUCB (split)                       ║
║    4. SFL Execution (forward / backward with Top-K, FedAvg)                ║
║    5. Physical Energy Accounting (Global Barrier T_global)                 ║
║    6. Agent Updates (rewards, replay buffer, covariance)                   ║
║                                                                            ║
║  Run:  python -m mta_sfl.main                                              ║
║  or:   python mta_sfl/main.py                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
import collections

import copy
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Ensure the project root is on the path so `mta_sfl.*` imports resolve
# regardless of the current working directory.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent          # mta_sfl/
_PROJECT_ROOT = _SCRIPT_DIR.parent                     # juhi_jha/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ═══════════════════════════════════════════════════════════════════════════════
# Module Imports  — mirror the directory layout exactly
# ═══════════════════════════════════════════════════════════════════════════════

# --- Config ---
from mta_sfl.config.system_config import SimulationConfig, default_config

# --- Physics ---
from mta_sfl.physics.mobility import (
    IoTDevice,
    UAVNode,
    CloudGateway,
    Entity3D,
    GaussMarkovMobility,
    initialize_entities,
    compute_route_lifetime,
)
from mta_sfl.physics.channel import (
    ChannelState,
    compute_all_sinr,
    ofdma_subchannel_bandwidth,
    shannon_rate,
)
from mta_sfl.physics.topology import TopologyManager
from mta_sfl.physics.energy import EnergyLedger, GlobalSyncBarrier

# --- ML Core ---
from mta_sfl.ml_core.compression import CompressionManager
from mta_sfl.ml_core.data_partitioner import DataPartitioner
from mta_sfl.ml_core.split_models import SplitModel
from mta_sfl.ml_core.orchestrator import SFLOrchestrator, ClusterConfig, RoundMetrics

# --- Agents ---
from mta_sfl.agents.ddqn_agent import DDQNAgent
from mta_sfl.agents.linucb_agent import (
    LinUCBSplitAllocator,
    LinUCBConfig,
    ContextExtractor,
)
from mta_sfl.agents.action_masking import DDQNActionMask, LinUCBActionMask
from mta_sfl.agents.cooperative_reward import CooperativeRewardComputer, RewardConfig
from mta_sfl.agents.constraint_logger import ConstraintLogger, RoundRecord
from mta_sfl.ablation_studies import AblationFlags, apply_no_topo_bridge_mask, apply_no_pseudo_count, get_dense_activation_bits

# --- Metrics ---
from mta_sfl.metrics.topology_tracker import TopologyStabilityTracker

# ═══════════════════════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mta_sfl.main")


# ═══════════════════════════════════════════════════════════════════════════════
# Crash-Resilient JSON Checkpoint  (§ 3)
# ═══════════════════════════════════════════════════════════════════════════════

def save_checkpoint_safe(data: Dict[str, Any], filepath: str) -> None:
    """Atomically write *data* as JSON.

    Writes to a temporary ``.tmp`` file first, then uses ``os.replace()``
    so that a crash mid-write will never corrupt the main checkpoint.
    """
    tmp_path = filepath + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp_path, filepath)


# ═══════════════════════════════════════════════════════════════════════════════
# Simulation Hyper-Parameters
# ═══════════════════════════════════════════════════════════════════════════════

NUM_IOTS   = 40
NUM_UAVS   = 16              # was 8 in config default — overridden here
NUM_CLOUD  = 1
NUM_ROUNDS = 150
# DATASET    = "fashion_mnist"  # or "cifar10"
DIRICHLET_ALPHA = 0.5
SEED       = 42
DEVICE     = "cpu"            # change to "cuda" if GPU is available
DATASETS: List[str] = [
    "fashion_mnist",
    "mnist",
    "cifar10",
    "cifar100",
    "uci_har",
]

# Smart GPU detection to prevent CPU bottlenecks on large datasets
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Top-K compression ratio (fraction of elements kept)
TOP_K_RATIO = 0.5

# FLOPs per model block — rough estimate for bookkeeping
FLOPS_PER_BLOCK = 1.0e6

# Safety caps to prevent T_global from blowing up when SINR is near-zero.
# In practice, a pipeline with a near-zero-rate link would be detected as
# infeasible by the action masks.  This cap prevents numerical overflow in
# the energy accounting for idle UAVs that still need to hover.
MIN_RATE_BPS     = 1.0e3         # 1 kbps floor on any Shannon link
MAX_COMM_DUR_S   = 10.0          # cap any single comm hop at 10 s

# Results directory
RESULTS_DIR = str(_PROJECT_ROOT / "results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "results.json")



# ═══════════════════════════════════════════════════════════════════════════════
# Helper: build tx-power vector for all N entities
# ═══════════════════════════════════════════════════════════════════════════════

def _build_tx_powers(cfg: SimulationConfig, N: int) -> np.ndarray:
    """Return a 1-D array of transmit powers (W) indexed by global node ID."""
    tx = np.zeros(N, dtype=np.float64)
    pwr = cfg.power
    tx[:cfg.num_iot] = pwr.P_tx_iot
    tx[cfg.num_iot : cfg.num_iot + cfg.num_uav] = pwr.P_tx_uav
    tx[cfg.num_iot + cfg.num_uav :] = pwr.P_tx_cloud
    return tx


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: Capacity-Aware OFDMA Association  (Step 2)
# ═══════════════════════════════════════════════════════════════════════════════

def ofdma_association(
    sinr_matrix: np.ndarray,
    iots: List[IoTDevice],
    uav_indices: List[int],
    energy_ledger: EnergyLedger,
    max_iots_per_uav: int = 5,
) -> Dict[int, List[int]]:
    """Assign each surviving IoT to the primary UAV with highest uplink SINR.

    Respects:
      - Capacity constraint:  each UAV serves ≤ ``max_iots_per_uav`` IoTs.
      - Battery liveness:     UAVs that are dead (``is_alive == False``) are
                              excluded.

    Args:
        sinr_matrix: Full N×N linear SINR matrix.
        iots: List of IoTDevice objects.
        uav_indices: Global node indices for all UAVs.
        energy_ledger: Current energy ledger.
        max_iots_per_uav: Maximum IoT clients per primary UAV.

    Returns:
        Mapping ``{uav_global_idx: [iot_global_idx, ...]}``.
    """
    associations: Dict[int, List[int]] = {u: [] for u in uav_indices}

    # Rank IoTs by their best-SINR UAV in descending order (greedy)
    iot_pref: List[Tuple[float, int, int]] = []   # (sinr, iot_idx, uav_idx)
    for iot in iots:
        i = iot.node_id
        if not energy_ledger.is_alive(i):
            continue
        for u in uav_indices:
            if not energy_ledger.is_alive(u):
                continue
            sinr_val = sinr_matrix[i, u]
            iot_pref.append((sinr_val, i, u))

    # Sort descending by SINR
    iot_pref.sort(key=lambda x: -x[0])

    assigned_iots: Set[int] = set()
    for sinr_val, iot_idx, uav_idx in iot_pref:
        if iot_idx in assigned_iots:
            continue
        if len(associations[uav_idx]) >= max_iots_per_uav:
            continue
        if sinr_val <= 0.0:
            continue
        associations[uav_idx].append(iot_idx)
        assigned_iots.add(iot_idx)

    # Remove empty clusters
    return {u: devs for u, devs in associations.items() if devs}


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: collect positions & node_types arrays from entity lists
# ═══════════════════════════════════════════════════════════════════════════════

def _collect_positions_and_types(
    iots: List[IoTDevice],
    uavs: List[UAVNode],
    clouds: List[CloudGateway],
) -> Tuple[np.ndarray, List[str]]:
    """Stack all entity positions into an (N, 3) array and matching type list."""
    entities: List[Entity3D] = list(iots) + list(uavs) + list(clouds)  # type: ignore[arg-type]
    positions = np.array([e.pos for e in entities], dtype=np.float64)
    node_types = [e.node_type for e in entities]
    return positions, node_types


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: compute Shannon rates for a specific pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def _pipeline_rates(
    sinr_matrix: np.ndarray,
    iot_idx: int,
    primary_uav_idx: int,
    relay_uav_idx: int,
    cloud_idx: int,
    cfg: SimulationConfig,
    cluster_size: int,
) -> Dict[str, float]:
    """Return Shannon-rate dict expected by LinUCBActionMask and ContextExtractor.

    Keys: ``R_iot_primary``, ``R_primary_relay``, ``R_relay_cloud``.
    """
    ch = cfg.channel
    bw_ul = ofdma_subchannel_bandwidth(ch.W_UL_hz, max(cluster_size, 1))
    sinr_ip = float(sinr_matrix[iot_idx, primary_uav_idx])
    
    if primary_uav_idx == relay_uav_idx:
        R_primary_relay = float("inf")
    else:
        sinr_pr = float(sinr_matrix[primary_uav_idx, relay_uav_idx])
        R_primary_relay = max(shannon_rate(ch.W_backhaul_hz, sinr_pr), MIN_RATE_BPS)
        
    sinr_rc = float(sinr_matrix[relay_uav_idx, cloud_idx])
    
    return {
        "R_iot_primary":   max(shannon_rate(bw_ul, sinr_ip), MIN_RATE_BPS),
        "R_primary_relay": R_primary_relay,
        "R_relay_cloud":   max(shannon_rate(ch.W_cloud_hz, sinr_rc), MIN_RATE_BPS),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def run_dataset_experiment(dataset_name: str, ablation_flags: Optional[AblationFlags] = None) -> Dict[str, Any]:
    """Execute Algorithm 1: Joint Execution Phase."""
    if ablation_flags is None:
        ablation_flags = AblationFlags()  # Full MTA-SFL by default

    # ------------------------------------------------------------------
    # 0.  Configuration
    # ------------------------------------------------------------------
    cfg = default_config()
    # Override with requested parameters
    cfg.num_iot   = NUM_IOTS
    cfg.num_uav   = NUM_UAVS
    cfg.num_cloud = NUM_CLOUD
    cfg.seed      = SEED

    N = cfg.num_iot + cfg.num_uav + cfg.num_cloud   # total entities
    rng = np.random.default_rng(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    dataset_result_file = os.path.join(RESULTS_DIR, f"results_{dataset_name}.json")
    constraint_log_path = os.path.join(RESULTS_DIR, f"constraint_log_{dataset_name}.json")
    
    logger.info("MTA-SFL  •  %d IoTs  •  %d UAVs  •  %d Rounds  •  %s  •  %s",
                cfg.num_iot, cfg.num_uav, NUM_ROUNDS, dataset_name.upper(), ablation_flags.name)

    # ------------------------------------------------------------------
    # 1.  Dataset Initialization  (out-of-the-box, no preprocessing)
    # ------------------------------------------------------------------
    logger.info("Initializing dataset: %s  (α=%.1f, mode=dirichlet)", dataset_name, DIRICHLET_ALPHA)
    data_partitioner = DataPartitioner(
        dataset_name=dataset_name,
        num_clients=cfg.num_iot,
        mode="dirichlet",
        alpha=DIRICHLET_ALPHA,
        batch_size=32,
        data_root=str(_PROJECT_ROOT / "data"),
        seed=cfg.seed,
    )
    client_loaders, test_loader = data_partitioner.partition()
    num_classes = data_partitioner.num_classes
    logger.info("Dataset ready: %d clients, %d classes", len(client_loaders), num_classes)

    # ------------------------------------------------------------------
    # 2.  Physical Entity Initialization
    # ------------------------------------------------------------------
    iots, uavs, clouds = initialize_entities(cfg, rng)
    all_entities: List[Entity3D] = list(iots) + list(uavs) + list(clouds)  # type: ignore[arg-type]

    # Index sets used throughout
    iot_indices: List[int]  = [d.node_id for d in iots]
    uav_indices: List[int]  = [u.node_id for u in uavs]
    cloud_indices: Set[int] = {c.node_id for c in clouds}
    cloud_idx: int          = clouds[0].node_id   # single cloud gateway

    logger.info("Entities: IoT [0..%d)  UAV [%d..%d)  Cloud [%d]",
                cfg.num_iot, cfg.num_iot, cfg.num_iot + cfg.num_uav, cloud_idx)

    # ------------------------------------------------------------------
    # 3.  Physical Layer Objects
    # ------------------------------------------------------------------
    # 3a. Mobility
    mobility = GaussMarkovMobility(cfg.mobility, seed=cfg.seed)

    # 3b. Channel
    positions, node_types = _collect_positions_and_types(iots, uavs, clouds)
    channel = ChannelState(positions, node_types, cfg, rng)

    # 3c. Topology
    topology = TopologyManager(cfg)

    # 3d. Energy
    energy_ledger = EnergyLedger(cfg.num_iot, cfg.num_uav, cfg.num_cloud, cfg)
    sync_barrier = GlobalSyncBarrier()

    # Transmit-power vector (constant across rounds)
    tx_powers = _build_tx_powers(cfg, N)

    # ------------------------------------------------------------------
    # 4.  ML / Agent Initialization
    # ------------------------------------------------------------------
    # 4a. Determine model depth L for the chosen dataset
    ref_model = SplitModel(dataset_name, num_classes)
    L = ref_model.L
    logger.info("SplitModel depth L = %d", L)

    # 4b. One DDQNAgent per UAV  (indexed by UAV global node ID)
    ddqn_agents: Dict[int, DDQNAgent] = {}
    for u_idx in uav_indices:
        ddqn_agents[u_idx] = DDQNAgent(
            num_uavs=cfg.num_uav,
            state_features_per_uav=5,
            lr=1e-3,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.995,
            target_update_freq=100,
            batch_size=64,
            buffer_capacity=50_000,
            beta_exploration=0.1,
            num_bins=10,
            device=DEVICE,
        )

    # 4c. DDQN Action Mask
    ddqn_mask_builder = DDQNActionMask(
        num_uavs=cfg.num_uav,
        uav_indices=uav_indices,
        cfg=cfg,
    )

    # 4d. LinUCB Split Allocator (shared across clusters)
    context_extractor = ContextExtractor(
        E_init_iot=cfg.battery.E_init_iot_J,
        E_init_uav=cfg.battery.E_init_uav_J,
        f_max=cfg.compute.f_cloud_hz,
        R_max=100.0e6,
    )
    linucb_config = LinUCBConfig(context_dim=9, alpha_ucb=1.5)
    linucb_allocator = LinUCBSplitAllocator(L, linucb_config, context_extractor)
    logger.info("LinUCB arm space: %d split configurations", linucb_allocator.get_num_arms())

    # 4e. LinUCB Action Mask
    linucb_mask_builder = LinUCBActionMask(L, cfg)

    # 4f. Cooperative Reward
    reward_config = RewardConfig()   # defaults: λ₁=0.4, λ₂=0.3, λ₃=0.3
    reward_computer = CooperativeRewardComputer(reward_config)

    # 4g. Constraint Logger & Topology Tracker
    constraint_logger = ConstraintLogger()
    topo_tracker = TopologyStabilityTracker()

    # ------------------------------------------------------------------
    # 5.  Persistent checkpoint state
    # ------------------------------------------------------------------
    checkpoint: Dict[str, Any] = {
        "config": {
            "num_iots": NUM_IOTS,
            "num_uavs": NUM_UAVS,
            "num_rounds": NUM_ROUNDS,
            "dataset": dataset_name,
            "dirichlet_alpha": DIRICHLET_ALPHA,
            "seed": SEED,
            "model_depth_L": L,
            "num_linucb_arms": linucb_allocator.get_num_arms(),
            "ablation": ablation_flags.name,
        },
        "rounds": [],
    }

    # ══════════════════════════════════════════════════════════════════
    #  ALGORITHM 1  —  Joint Execution Loop  (t = 1 … T)
    # ══════════════════════════════════════════════════════════════════
    global_model_state_dict = copy.deepcopy(ref_model.state_dict())
    global_momentum = {k: torch.zeros_like(v) for k, v in global_model_state_dict.items()}
    global_compression_manager = CompressionManager(TOP_K_RATIO)
    ddqn_transition_cache = {}  # {primary_idx: (state, action, reward)}
    total_latency_s: float = 0.0

    logger.info("═" * 60)
    logger.info("  Algorithm 1 — Starting %d rounds", NUM_ROUNDS)
    logger.info("═" * 60)

    for t in range(1, NUM_ROUNDS + 1):
        round_start_wall = time.perf_counter()

        # ==============================================================
        # STEP 1 — Mobility & Channel Update
        # ==============================================================
        # 1a. Step Gauss-Markov mobility for all UAVs
        mobility.step(uavs, cfg.dt_s)

        # 1b. Recompute positions array and update channel gains
        positions, _ = _collect_positions_and_types(iots, uavs, clouds)
        channel.update(positions)

        # 1c. Compute initial associations (needed for OFDMA interference)
        #     Use a preliminary greedy assignment so compute_all_sinr can
        #     model inter-cluster interference correctly.
        preliminary_assoc: Dict[int, List[int]] = {}
        for u_idx in uav_indices:
            preliminary_assoc[u_idx] = []
        for iot in iots:
            if not energy_ledger.is_alive(iot.node_id):
                continue
            best_u = max(uav_indices, key=lambda u: channel.gain_matrix[iot.node_id, u])
            preliminary_assoc.setdefault(best_u, []).append(iot.node_id)

        # 1d. Full N×N SINR matrix (OFDMA-aware)
        sinr_matrix = compute_all_sinr(channel, preliminary_assoc, tx_powers, cfg)

        # 1e. Update topology adjacency from SINR
        topology.update(sinr_matrix)

        # ==============================================================
        # STEP 2 — Capacity-Aware OFDMA Association
        # ==============================================================
        associations = ofdma_association(
            sinr_matrix, iots, uav_indices, energy_ledger,
            max_iots_per_uav=5,
        )
        active_primaries = sorted(associations.keys())
        num_active_iots = sum(len(v) for v in associations.values())

        # If no associations this round (extreme energy depletion), skip
        if not active_primaries:
            logger.warning("[Round %d/%d] No active clusters — skipping", t, NUM_ROUNDS)
            continue

        # Re-compute SINR with correct cluster assignments
        sinr_matrix = compute_all_sinr(channel, associations, tx_powers, cfg)
        topology.update(sinr_matrix)

        # ==============================================================
        # STEP 3 — Action Masking  →  DDQN (relay)  →  LinUCB (split)
        # ==============================================================
        relay_selections: Dict[int, int]                = {}   # primary → relay (global idx)
        split_selections: Dict[int, Tuple[int,int,int]] = {}   # primary → (p1,p2,p3)
        cluster_configs: List[ClusterConfig]             = []

        # Per-cluster tracking for reward/energy accounting
        linucb_arm_indices: Dict[int, int]      = {}  # primary → arm index
        linucb_contexts:    Dict[int, np.ndarray] = {}
        ddqn_states:        Dict[int, np.ndarray] = {}
        ddqn_masks:         Dict[int, np.ndarray] = {}
        route_failed_flags: Dict[int, bool]     = {}

        cloud_reachable: Set[int] = topology.cloud_reachable_set(cloud_indices)
        T_max = reward_config.T_max
        
        # Track energy commitments across clusters in the same round to prevent relay over-commitment (Bug 4)
        relay_energy_commitments = collections.defaultdict(float)

        for primary_idx in active_primaries:
            iot_list = associations[primary_idx]
            # The local UAV index within uav_indices for DDQN (0-based)
            primary_local = uav_indices.index(primary_idx)

            # ----------------------------------------------------------
            # 3a. DDQN state construction
            # ----------------------------------------------------------
            cc_vals = np.array([
                topology.get_clustering_coefficient(u) for u in uav_indices
            ])
            e_fracs = np.array([
                energy_ledger.get_fraction(u) for u in uav_indices
            ])
            sinr_vals = np.array([
                float(sinr_matrix[primary_idx, u]) if u != primary_idx else 0.0
                for u in uav_indices
            ])
            lifetimes = np.array([
                compute_route_lifetime(
                    all_entities[primary_idx], all_entities[u], cfg.mobility.R_comm_m
                ) if u != primary_idx else 0.0
                for u in uav_indices
            ])
            q_loads = np.array([
                topology.queue.get_load(u) for u in uav_indices
            ])

            state = ddqn_agents[primary_idx].build_state(
                cc_vals, e_fracs, sinr_vals, lifetimes, q_loads
            )
            ddqn_states[primary_idx] = state

            # ----------------------------------------------------------
            # 3b. DDQN action mask
            # ----------------------------------------------------------
            mask, is_routing_failure = ddqn_mask_builder.compute_mask(
                primary_uav_idx=primary_idx,
                sinr_matrix=sinr_matrix,
                energy_ledger=energy_ledger,
                entities=all_entities,
                cloud_indices=cloud_indices,
                topology=topology,
                T_max=T_max,
            )
            ddqn_masks[primary_idx] = mask
            route_failed_flags[primary_idx] = is_routing_failure

            # [FIX] Push cached transition from round t-1 using the TRUE next_state and next_mask
            if primary_idx in ddqn_transition_cache:
                prev_state, prev_action, prev_reward = ddqn_transition_cache[primary_idx]
                ddqn_agents[primary_idx].store_transition(
                    prev_state, prev_action, prev_reward, state, mask, done=False
                )

            # ----------------------------------------------------------
            # 3c. DDQN relay selection
            # ----------------------------------------------------------
            if is_routing_failure:
                # No feasible relay — use self-relay (bridge mode forced)
                relay_local = primary_local
                relay_global = primary_idx
            else:
                relay_local, _ = ddqn_agents[primary_idx].select_action(state, mask)
                relay_global = uav_indices[relay_local]

            relay_selections[primary_idx] = relay_global

            # ----------------------------------------------------------
            # 3d. LinUCB action mask (split feasibility)
            # ----------------------------------------------------------
            #     Use the first IoT in the cluster for representative rates
            rep_iot = iot_list[0]
            cluster_size = len(iot_list)
            rates = _pipeline_rates(
                sinr_matrix, rep_iot, primary_idx, relay_global,
                cloud_idx, cfg, cluster_size,
            )

            linucb_mask, linucb_violations = linucb_mask_builder.compute_mask(
                arms=linucb_allocator.arm_list,
                iot_idx=rep_iot,
                primary_uav_idx=primary_idx,
                relay_uav_idx=relay_global,
                energy_ledger=energy_ledger,
                rates=rates,
                T_max=T_max,
                flops_per_block=FLOPS_PER_BLOCK,
                cluster_size=cluster_size,
                reserved_energy_map=relay_energy_commitments,
            )

            # --- Ablation: No Topological Bridge ---
            if not ablation_flags.use_topo_bridge:
                linucb_mask = apply_no_topo_bridge_mask(linucb_mask, linucb_allocator.arm_list)

            # ----------------------------------------------------------
            # 3e. LinUCB context construction & split selection
            # ----------------------------------------------------------
            ctx = context_extractor.extract_and_normalize(
                E_res_iot=energy_ledger.get_residual(rep_iot),
                E_res_primary=energy_ledger.get_residual(primary_idx),
                E_res_relay=energy_ledger.get_residual(relay_global),
                f_iot=cfg.compute.f_iot_hz,
                f_primary=cfg.compute.f_uav_hz,
                f_relay=cfg.compute.f_uav_hz,
                R_iot_primary=rates["R_iot_primary"],
                R_primary_relay=rates["R_primary_relay"],
                R_relay_cloud=rates["R_relay_cloud"],
            )
            linucb_contexts[primary_idx] = ctx

            if not np.any(linucb_mask):
                # No feasible split — fall back to even split (L//3, 2*L//3, L)
                split_cfg = (max(1, L // 3), max(1, 2 * L // 3), L)
                arm_idx = -1
            else:
                split_cfg, arm_idx, _ = linucb_allocator.select_split(ctx, linucb_mask)

            linucb_arm_indices[primary_idx] = arm_idx
            split_selections[primary_idx] = split_cfg
            
            # Bug 4 Fix: Dynamically reserve the estimated relay energy to prevent over-commitment
            from mta_sfl.agents.action_masking import estimate_pipeline_energy
            E_est = estimate_pipeline_energy(
                p=split_cfg,
                L=L,
                cfg=cfg,
                rates=rates,
                flops_per_block=FLOPS_PER_BLOCK,
                cluster_size=cluster_size
            )
            relay_energy_commitments[relay_global] += E_est["relay"]

            # ----------------------------------------------------------
            # 3f. Build ClusterConfig for the orchestrator
            # ----------------------------------------------------------
            cluster_configs.append(ClusterConfig(
                cluster_id=f"cluster_{primary_idx}",
                primary_uav_id=str(primary_idx),
                relay_uav_id=str(relay_global) if relay_global != primary_idx else None,
                iot_ids=[str(i) for i in iot_list],
                cut_config=split_cfg,
            ))

            # ----------------------------------------------------------
            # 3g. Log topology tracking
            # ----------------------------------------------------------
            is_bridge = (split_cfg[1] == split_cfg[2])
            topo_tracker.log_round(primary_idx, relay_global, is_bridge)

        topo_tracker.step_round()

        # ==============================================================
        # STEP 4 — SFL Execution
        # ==============================================================
        # Reset queues from prev round
        topology.queue.reset()
        # Build the orchestrator with this round's cluster configs
        k_ratio_effective = TOP_K_RATIO if ablation_flags.use_topk else 1.0
        orchestrator = SFLOrchestrator(
            dataset_name=dataset_name,
            num_classes=num_classes,
            clusters=cluster_configs,
            k_ratio=k_ratio_effective,
            lr=0.01,
            optimizer_name="sgd",
            momentum=0.0,  # Local momentum disabled in favor of Server Momentum (FedAvgM)
            device=DEVICE,
            global_state_dict=global_model_state_dict,
            global_compression_manager=global_compression_manager,
        )

        # Prepare iot_loaders dict keyed by string IDs
        iot_loaders: Dict[str, Any] = {}
        for primary_idx, iot_list in associations.items():
            for iot_id in iot_list:
                iot_loaders[str(iot_id)] = client_loaders[iot_id]

        # Execute one SFL round
        round_metrics: RoundMetrics = orchestrator.run_round(
            round_idx=t,
            iot_loaders=iot_loaders,
            test_loader=test_loader,
        )

        # Extract the new aggregated global model state
        new_state_dict = orchestrator.assemble_global_model().state_dict()
        
        # Apply Server Momentum (FedAvgM) to persist training dynamics across changing topologies
        for k in global_model_state_dict.keys():
            if "weight" in k or "bias" in k:
                pseudo_grad = global_model_state_dict[k] - new_state_dict[k]
                global_momentum[k] = 0.9 * global_momentum[k] + pseudo_grad
                global_model_state_dict[k] -= global_momentum[k]
            else:
                global_model_state_dict[k] = new_state_dict[k]

        # ==============================================================
        # STEP 5 — Physical Energy Accounting (Global Barrier)
        # ==============================================================
        # 5a. Estimate per-pipeline task durations from metrics
        #     The orchestrator's train_loss is aggregated; for per-pipeline
        #     latency we derive from a simplified computation model.
        task_durations: Dict[int, float] = {}
        rf_durations:   Dict[int, Tuple[float, float, float, float]] = {}
        comp_flops:     Dict[int, float] = {}
        node_types_map: Dict[int, str]   = {}
        t_pipeline_map: Dict[int, float] = {}

        # All involved nodes participate for the duration
        for primary_idx, iot_list in associations.items():
            relay_idx = relay_selections[primary_idx]
            split = split_selections[primary_idx]
            p1, p2, p3 = split
            cluster_size = len(iot_list)

            # Representative pipeline rates for duration estimation
            rep_iot = iot_list[0]
            rates = _pipeline_rates(
                sinr_matrix, rep_iot, primary_idx, relay_idx,
                cloud_idx, cfg, cluster_size,
            )

            # Computation durations per tier
            flops_iot   = p1 * FLOPS_PER_BLOCK * cluster_size
            flops_prim  = (p2 - p1) * FLOPS_PER_BLOCK * cluster_size
            flops_relay = (p3 - p2) * FLOPS_PER_BLOCK * cluster_size
            flops_cloud = (L - p3) * FLOPS_PER_BLOCK * cluster_size

            t_comp_iot   = flops_iot / cfg.compute.f_iot_hz
            t_comp_prim  = flops_prim / cfg.compute.f_uav_hz
            t_comp_relay = flops_relay / cfg.compute.f_uav_hz
            t_comp_cloud = flops_cloud / cfg.compute.f_cloud_hz

            # Communication durations (1 Mbit intermediate activations per link)
            # Capped at MAX_COMM_DUR_S — in practice a link this slow would
            # have been masked as infeasible, but idle UAVs still hover.
            activation_bits = 1.0e6 * TOP_K_RATIO if ablation_flags.use_topk else 1.0e6
            t_comm_ip = min(activation_bits / max(rates["R_iot_primary"], MIN_RATE_BPS),
                            MAX_COMM_DUR_S)
            t_comm_pr = 0.0 if relay_idx == primary_idx else min(
                activation_bits / max(rates["R_primary_relay"], MIN_RATE_BPS),
                MAX_COMM_DUR_S
            )
            t_comm_rc = min(activation_bits / max(rates["R_relay_cloud"], MIN_RATE_BPS),
                            MAX_COMM_DUR_S)

            t_pipeline = (t_comp_iot + t_comm_ip + t_comp_prim + t_comm_pr
                          + t_comp_relay + t_comm_rc + t_comp_cloud)
            t_pipeline_map[primary_idx] = t_pipeline

            # IoT devices
            for iot_id in iot_list:
                task_durations[iot_id] = t_comp_iot / cluster_size + t_comm_ip
                rf_durations[iot_id] = (cfg.power.P_tx_iot, t_comm_ip,
                                        cfg.power.P_rx_iot, t_comm_ip)
                comp_flops[iot_id] = p1 * FLOPS_PER_BLOCK
                node_types_map[iot_id] = "iot"

            # Primary UAV
            task_durations[primary_idx] = task_durations.get(primary_idx, 0.0) + t_comp_prim + t_comm_ip + t_comm_pr
            prev_rf = rf_durations.get(primary_idx, (cfg.power.P_tx_uav, 0.0, cfg.power.P_rx_uav, 0.0))
            rf_durations[primary_idx] = (
                cfg.power.P_tx_uav, prev_rf[1] + t_comm_pr,
                cfg.power.P_rx_uav, prev_rf[3] + t_comm_ip
            )
            comp_flops[primary_idx] = comp_flops.get(primary_idx, 0.0) + (p2 - p1) * FLOPS_PER_BLOCK * cluster_size
            node_types_map[primary_idx] = "uav"
            
            # Record congestion for the RL agent's next observation (simulate queueing)
            # We enqueue the workload load now, and it will be visible in the next round.
            topology.queue.enqueue(primary_idx, activation_bits * cluster_size)

            # Relay UAV
            if relay_idx != primary_idx:
                task_durations[relay_idx] = task_durations.get(relay_idx, 0.0) + t_comp_relay + t_comm_pr + t_comm_rc
                prev_rf = rf_durations.get(relay_idx, (cfg.power.P_tx_uav, 0.0, cfg.power.P_rx_uav, 0.0))
                rf_durations[relay_idx] = (
                    cfg.power.P_tx_uav, prev_rf[1] + t_comm_rc,
                    cfg.power.P_rx_uav, prev_rf[3] + t_comm_pr
                )
                comp_flops[relay_idx] = comp_flops.get(relay_idx, 0.0) + (p3 - p2) * FLOPS_PER_BLOCK * cluster_size
                node_types_map[relay_idx] = "uav"
                topology.queue.enqueue(relay_idx, activation_bits * cluster_size)
            else:
                # Primary is also the relay; charge it the cloud transmission and relay compute
                task_durations[primary_idx] += t_comp_relay + t_comm_rc
                prev_rf = rf_durations[primary_idx]
                rf_durations[primary_idx] = (
                    cfg.power.P_tx_uav, prev_rf[1] + t_comm_rc,
                    cfg.power.P_rx_uav, prev_rf[3]
                )
                comp_flops[primary_idx] += (p3 - p2) * FLOPS_PER_BLOCK * cluster_size

        # Cloud
        task_durations[cloud_idx] = sum(
            (L - split_selections[p][2]) * FLOPS_PER_BLOCK
            * len(associations[p]) / cfg.compute.f_cloud_hz
            for p in active_primaries
        )
        rf_durations[cloud_idx] = (cfg.power.P_tx_cloud, 0.0,
                                    cfg.power.P_rx_cloud, 0.0)
        comp_flops[cloud_idx] = sum(
            (L - split_selections[p][2]) * FLOPS_PER_BLOCK * len(associations[p])
            for p in active_primaries
        )
        node_types_map[cloud_idx] = "cloud"

        # Also add any UAVs that are alive but idle (they still hover)
        for u_idx in uav_indices:
            if u_idx not in node_types_map and energy_ledger.is_alive(u_idx):
                task_durations[u_idx] = 0.0
                rf_durations[u_idx] = (0.0, 0.0, 0.0, 0.0)
                comp_flops[u_idx] = 0.0
                node_types_map[u_idx] = "uav"

        # 5b. Compute T_global = max over all task durations
        T_global = sync_barrier.compute_T_global(task_durations)

        # CAPTURE PRE-DEDUCTION RESIDUALS FOR REWARDS (Bug 1 Fix)
        # Prevents e_res from being 0.0 which explodes the reward function.
        pre_deduction_residuals = {n: energy_ledger.get_residual(n) for n in range(len(all_entities))}

        # 5c. Apply energy deductions
        round_energies = energy_ledger.apply_global_round(
            T_global=T_global,
            task_durations=task_durations,
            rf_durations=rf_durations,
            comp_flops=comp_flops,
            node_types=node_types_map,
            cfg=cfg,
        )

        # ==============================================================
        # STEP 6 — Agent Updates (Rewards, Replay, Covariance)
        # ==============================================================
        avg_ddqn_reward = 0.0
        num_rewarded = 0

        for primary_idx in active_primaries:
            relay_idx = relay_selections[primary_idx]
            split = split_selections[primary_idx]
            iot_list = associations[primary_idx]

            # Per-pipeline residuals and energy consumed
            pipeline_nodes = set(iot_list + [primary_idx, relay_idx])
            per_node_energy = {n: round_energies.get(n, 0.0) for n in pipeline_nodes}
            per_node_residual = {n: pre_deduction_residuals[n] for n in pipeline_nodes}

            # Clustering coefficient of relay
            cc_relay = topology.get_clustering_coefficient(relay_idx)

            # Shadow energy strictly bounded to local pipeline time (isolates agents from global stragglers)
            t_pipe = t_pipeline_map[primary_idx]
            per_node_energy_shadow = {}
            for n in pipeline_nodes:
                ntype = node_types_map[n]
                rf = rf_durations.get(n, (0.0, 0.0, 0.0, 0.0))
                flops = comp_flops.get(n, 0.0)
                e_rf = rf[0] * rf[1] + rf[2] * rf[3]
                if ntype == "uav":
                    e_prop = cfg.power.P_flight_hover * t_pipe
                    e_cir = cfg.power.P_circuit_uav * t_pipe
                    e_comp = cfg.compute.kappa * (cfg.compute.f_uav_hz**2) * flops
                    total = e_prop + e_cir + e_rf + e_comp
                elif ntype == "iot":
                    t_act = min(task_durations.get(n, 0.0), t_pipe)
                    e_cir = cfg.power.P_circuit_iot_active * t_act + cfg.power.P_sleep_iot * max(0.0, t_pipe - t_act)
                    e_comp = cfg.compute.kappa * (cfg.compute.f_iot_hz**2) * flops
                    total = e_cir + e_rf + e_comp
                else:
                    total = 0.0
                per_node_energy_shadow[n] = total

            # Constraint violation check
            constraint_violated = (t_pipe > T_max)

            # ---- 6a. DDQN reward ----
            r_ext = reward_computer.compute_ddqn_reward(
                T_total=t_pipe,
                per_node_energy=per_node_energy_shadow,
                per_node_residual=per_node_residual,
                cc_relay=cc_relay,
                constraint_violated=constraint_violated,
                route_failed=route_failed_flags.get(primary_idx, False),
            )

            # Exploration bonus
            state = ddqn_states[primary_idx]
            relay_local = uav_indices.index(relay_idx)
            r_int = ddqn_agents[primary_idx].get_exploration_bonus(state, relay_local)
            # --- Ablation: No Pseudo-Count ---
            if not ablation_flags.use_pseudo_count:
                r_int = apply_no_pseudo_count(r_int)
            r_total = r_ext + r_int

            avg_ddqn_reward += r_ext
            num_rewarded += 1

            # ---- 6b. Cache DDQN transition (will be stored at t+1) ----
            if not is_bridge:
                ddqn_transition_cache[primary_idx] = (state, relay_local, r_total)

                # If this is the final round, immediately push to flush cache
                if t == NUM_ROUNDS:
                    next_mask = ddqn_masks.get(primary_idx, np.ones(cfg.num_uav, dtype=bool))
                    ddqn_agents[primary_idx].store_transition(
                        state, relay_local, r_total, state, next_mask, done=True
                    )

            # ---- 6c. DDQN training step ----
            ddqn_agents[primary_idx].train_step()
            
            # Note: store_transition() in ddqn_agent.py already increments _step_count 
            # and triggers update_target_network(), so we do not double-call it here.


            # ---- 6d. LinUCB reward & update ----

                
            r_linucb = reward_computer.compute_linucb_reward(
                T_total=t_pipe,
                per_node_energy=per_node_energy_shadow,
                per_node_residual=per_node_residual,
            )
            arm_idx = linucb_arm_indices.get(primary_idx, -1)
            ctx = linucb_contexts.get(primary_idx)
            if arm_idx >= 0 and ctx is not None:
                linucb_allocator.update_with_reward(arm_idx, ctx, r_linucb)

            # ---- 6e. Log constraint record ----
            ddqn_mask = ddqn_masks.get(primary_idx, np.zeros(cfg.num_uav, dtype=bool))
            record = RoundRecord(
                round_idx=t,
                primary_uav_idx=primary_idx,
                selected_relay=relay_idx,
                selected_split=split,
                ddqn_feasible_count=int(np.sum(ddqn_mask)),
                linucb_feasible_count=int(np.sum(
                    linucb_mask_builder.compute_mask(
                        linucb_allocator.arm_list,
                        iot_list[0], primary_idx, relay_idx,
                        energy_ledger,
                        _pipeline_rates(sinr_matrix, iot_list[0], primary_idx,
                                        relay_idx, cloud_idx, cfg, len(iot_list)),
                        T_max, FLOPS_PER_BLOCK, len(iot_list)
                    )[0]
                )),
                constraint_compliant=not constraint_violated,
                r_ext=r_ext,
                r_int=r_int,
                r_total=r_total,
                is_bridge_mode=(split[1] == split[2]),
                bridge_forced=route_failed_flags.get(primary_idx, False),
                epsilon=ddqn_agents[primary_idx].epsilon,
                T_total=T_global,
                E_ratio=reward_computer.compute_energy_ratio(per_node_energy, per_node_residual),
                relay_battery_fraction=energy_ledger.get_fraction(relay_idx),
            )
            constraint_logger.log_round(record)

        # ---- 6f. Decay epsilon for all active DDQN agents ----
        for primary_idx in active_primaries:
            ddqn_agents[primary_idx].decay_epsilon()

        # ==============================================================
        # STEP 7 — Checkpoint & Terminal Output
        # ==============================================================
        total_latency_s += T_global
        
        # Only log every 10 rounds or on the final round to save memory on 5000-round runs
        if t % 10 == 0 or t == NUM_ROUNDS:
            # Representative epsilon (first active primary)
            eps = ddqn_agents[active_primaries[0]].epsilon if active_primaries else 1.0
            energy_summary = energy_ledger.summary()
            cumulative_energy = float(np.sum(energy_ledger.E_initial) - np.sum(energy_ledger.E_residual))

            round_telemetry = {
                "round": t,
                "test_accuracy": round(round_metrics.test_accuracy * 100, 2),
                "test_loss": round(round_metrics.test_loss, 4),
                "train_loss": round(round_metrics.train_loss, 4),
                "avg_latency_s": round(T_global, 4),
                "cumulative_energy_J": round(cumulative_energy, 4),
                "num_active_iots": num_active_iots,
                "total_iots": cfg.num_iot,
                "num_active_clusters": len(active_primaries),
                "epsilon": round(eps, 4),
                "min_energy_fraction": round(energy_summary["min_fraction"], 4),
                "avg_energy_fraction": round(energy_summary["avg_fraction"], 4),
                "num_alive_nodes": energy_summary["num_alive"],
                "compliance_rate": round(constraint_logger.get_compliance_rate(window=50), 4),
                "churn_rate": round(topo_tracker.get_churn_rate(), 4),
                "wall_time_s": round(time.perf_counter() - round_start_wall, 2),
            }
            checkpoint["rounds"].append(round_telemetry)
            save_checkpoint_safe(checkpoint, dataset_result_file)

        # Clean terminal output
        acc = round_metrics.test_accuracy * 100
        loss = round_metrics.test_loss
        print(
            f"[Round {t:3d}/{NUM_ROUNDS}] "
            f"Acc: {acc:5.1f}% | "
            f"Loss: {loss:.3f} | "
            f"Latency: {T_global:.2f}s | "
            f"IoTs: {num_active_iots}/{cfg.num_iot} | "
            f"Eps: {eps:.2f}"
        )

    # ══════════════════════════════════════════════════════════════════
    #  Post-Simulation Summary
    # ══════════════════════════════════════════════════════════════════
    logger.info("═" * 60)
    logger.info("  Simulation Complete — %d rounds", NUM_ROUNDS)
    logger.info("═" * 60)

    final_acc = checkpoint["rounds"][-1]["test_accuracy"] if checkpoint["rounds"] else 0.0
    logger.info("  Final Test Accuracy : %.2f%%", final_acc)
    logger.info("  Compliance Rate     : %.2f%%",
                constraint_logger.get_compliance_rate() * 100)
    logger.info("  Relay Churn Rate    : %.4f", topo_tracker.get_churn_rate())
    logger.info("  Energy (min frac)   : %.4f", energy_ledger.summary()["min_fraction"])
    logger.info("  Results saved to    : %s", dataset_result_file)

    # Export constraint log
    constraint_log_path = os.path.join(RESULTS_DIR, "constraint_log.json")
    constraint_logger.export_json(constraint_log_path)
    logger.info("  Constraint log      : %s", constraint_log_path)
    rounds = checkpoint.get("rounds", [])
    avg_latency = total_latency_s / NUM_ROUNDS if NUM_ROUNDS > 0 else 0.0
    uav_indices = [u.node_id for u in uavs]
    drones_depleted = sum(1 for u in uav_indices if not energy_ledger.is_alive(u))

    final_stats = {
        "dataset": dataset_name,
        "final_accuracy": final_acc,
        "final_loss": rounds[-1]["test_loss"] if rounds else 0.0,
        "avg_latency_s": round(avg_latency, 4),
        "total_energy_J": float(np.sum(energy_ledger.E_initial) - np.sum(energy_ledger.E_residual)),
        "drones_depleted": drones_depleted,
        "compliance_rate": constraint_logger.get_compliance_rate() * 100,
        "churn_rate": topo_tracker.get_churn_rate(),
        "min_energy_frac": energy_ledger.summary()["min_fraction"],
    }

    # CRITICAL: Free memory between dataset runs to prevent OOM crashes
    import gc
    del ddqn_agents, linucb_allocator, orchestrator, client_loaders, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return final_stats


# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# Main Sequential Loop
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    master_summary: List[Dict[str, Any]] = []

    logger.info("Starting sequential execution across %d datasets: %s", len(DATASETS), DATASETS)
    overall_start = time.perf_counter()

    for ds in DATASETS:
        try:
            stats = run_dataset_experiment(ds)
            master_summary.append(stats)
            
            # Atomic update of the master summary file after each dataset completes
            save_checkpoint_safe(
                {"datasets": master_summary, "total_wall_s": time.perf_counter() - overall_start},
                os.path.join(RESULTS_DIR, "all_datasets_summary.json"),
            )
        except Exception as exc:
            logger.error("Error executing dataset '%s': %s", ds, exc, exc_info=True)
            logger.info("Proceeding to next dataset in queue...")

    logger.info("═" * 70)
    logger.info("ALL DATASET RUNS COMPLETE in %.2f minutes", (time.perf_counter() - overall_start) / 60)
    logger.info("═" * 70)
    
    for res in master_summary:
        logger.info(
            "%-14s │ Acc: %5.2f%% │ Loss: %6.4f │ Compliance: %5.1f%% │ Min Battery: %.2f",
            res["dataset"].upper(), res["final_accuracy"], res["final_loss"],
            res["compliance_rate"], res["min_energy_frac"]
        )

if __name__ == "__main__":
    main()
