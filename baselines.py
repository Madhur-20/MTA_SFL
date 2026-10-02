import json
import logging
import os
import time
import copy
from typing import Dict, List, Set, Tuple, Any, Optional

import numpy as np
import networkx as nx
import torch

# --- Config ---
from mta_sfl.config.system_config import SimulationConfig, default_config

# --- Physics ---
from mta_sfl.physics.mobility import (
    IoTDevice, UAVNode, CloudGateway, Entity3D, 
    GaussMarkovMobility, initialize_entities
)
from mta_sfl.physics.channel import (
    ChannelState, compute_all_sinr, 
    ofdma_subchannel_bandwidth, shannon_rate
)
from mta_sfl.physics.topology import TopologyManager
from mta_sfl.physics.energy import EnergyLedger, GlobalSyncBarrier

# --- ML Core ---
from mta_sfl.ml_core.compression import CompressionManager
from mta_sfl.ml_core.data_partitioner import DataPartitioner
from mta_sfl.ml_core.split_models import SplitModel
from mta_sfl.ml_core.orchestrator import SFLOrchestrator, ClusterConfig

# --- Metrics ---
from mta_sfl.metrics.episode_metrics import EpisodeMetrics

logger = logging.getLogger("mta_sfl.baselines")

# Constants matching main.py
NUM_IOTS = 40
NUM_UAVS = 16
NUM_CLOUD = 1
NUM_ROUNDS = 150
DIRICHLET_ALPHA = 0.5
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K_RATIO = 0.5
FLOPS_PER_BLOCK = 1.0e6
MIN_RATE_BPS = 1.0e3
MAX_COMM_DUR_S = 10.0

def _build_tx_powers(cfg: SimulationConfig, N: int) -> np.ndarray:
    tx = np.zeros(N, dtype=np.float64)
    pwr = cfg.power
    tx[:cfg.num_iot] = pwr.P_tx_iot
    tx[cfg.num_iot : cfg.num_iot + cfg.num_uav] = pwr.P_tx_uav
    tx[cfg.num_iot + cfg.num_uav :] = pwr.P_tx_cloud
    return tx

def _collect_positions_and_types(iots, uavs, clouds) -> Tuple[np.ndarray, List[str]]:
    entities = list(iots) + list(uavs) + list(clouds)
    positions = np.array([e.pos for e in entities], dtype=np.float64)
    node_types = [e.node_type for e in entities]
    return positions, node_types

def ofdma_association(sinr_matrix, iots, uav_indices, energy_ledger, max_iots_per_uav=5) -> Dict[int, List[int]]:
    associations = {u: [] for u in uav_indices}
    iot_pref = []
    for iot in iots:
        i = iot.node_id
        if not energy_ledger.is_alive(i):
            continue
        for u in uav_indices:
            if not energy_ledger.is_alive(u):
                continue
            iot_pref.append((sinr_matrix[i, u], i, u))
    
    iot_pref.sort(key=lambda x: -x[0])
    assigned_iots = set()
    for sinr_val, iot_idx, uav_idx in iot_pref:
        if iot_idx in assigned_iots: continue
        if len(associations[uav_idx]) >= max_iots_per_uav: continue
        if sinr_val <= 0.0: continue
        associations[uav_idx].append(iot_idx)
        assigned_iots.add(iot_idx)
        
    return {u: devs for u, devs in associations.items() if devs}

def _pipeline_rates(sinr_matrix, iot_idx, primary_uav_idx, relay_uav_idx, cloud_idx, cfg, cluster_size) -> Dict[str, float]:
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
        "R_iot_primary": max(shannon_rate(bw_ul, sinr_ip), MIN_RATE_BPS),
        "R_primary_relay": R_primary_relay,
        "R_relay_cloud": max(shannon_rate(ch.W_cloud_hz, sinr_rc), MIN_RATE_BPS),
    }

def estimate_pipeline_cost(primary_idx, relay_idx, l, iot_list, sinr_matrix, cfg, cloud_idx, L):
    # Returns (energy, latency) for a single pipeline choice
    # l is the cut layer for primary UAV. Split is (1, l, l)
    cluster_size = len(iot_list)
    rep_iot = iot_list[0]
    rates = _pipeline_rates(sinr_matrix, rep_iot, primary_idx, relay_idx, cloud_idx, cfg, cluster_size)
    
    p1 = 1
    p2 = l
    p3 = l
    
    flops_iot = p1 * FLOPS_PER_BLOCK * cluster_size
    flops_prim = (p2 - p1) * FLOPS_PER_BLOCK * cluster_size
    flops_relay = (p3 - p2) * FLOPS_PER_BLOCK * cluster_size
    flops_cloud = (L - p3) * FLOPS_PER_BLOCK * cluster_size
    
    t_comp_iot = flops_iot / cfg.compute.f_iot_hz
    t_comp_prim = flops_prim / cfg.compute.f_uav_hz
    t_comp_relay = flops_relay / cfg.compute.f_uav_hz
    t_comp_cloud = flops_cloud / cfg.compute.f_cloud_hz
    
    activation_bits = 1.0e6
    t_comm_ip = min(activation_bits / max(rates["R_iot_primary"], MIN_RATE_BPS), MAX_COMM_DUR_S)
    t_comm_pr = 0.0 if relay_idx == primary_idx else min(activation_bits / max(rates["R_primary_relay"], MIN_RATE_BPS), MAX_COMM_DUR_S)
    t_comm_rc = min(activation_bits / max(rates["R_relay_cloud"], MIN_RATE_BPS), MAX_COMM_DUR_S) if p3 < L else 0.0
    
    t_pipeline = (t_comp_iot / cluster_size + t_comm_ip + 
                  t_comp_prim + t_comm_pr + 
                  t_comp_relay + t_comm_rc + 
                  t_comp_cloud)
    
    # Rough energy estimation
    e_iot = cluster_size * (t_comp_iot/cluster_size * cfg.power.P_circuit_iot_active + t_comm_ip * cfg.power.P_tx_iot)
    e_prim = t_comp_prim * cfg.power.P_circuit_uav + t_comm_pr * cfg.power.P_tx_uav + t_comm_ip * cfg.power.P_rx_uav + t_pipeline * cfg.power.P_flight_hover
    if relay_idx == primary_idx:
        e_prim += t_comp_relay * cfg.power.P_circuit_uav + t_comm_rc * cfg.power.P_tx_uav
    e_relay = t_comp_relay * cfg.power.P_circuit_uav + t_comm_rc * cfg.power.P_tx_uav + t_comm_pr * cfg.power.P_rx_uav + t_pipeline * cfg.power.P_flight_hover if relay_idx != primary_idx else 0
    e_total = e_iot + e_prim + e_relay
    
    return e_total, t_pipeline


# ═══════════════════════════════════════════════════════════════════════
# AdaptSFL Paper-Faithful Helpers
# Lin et al., "AdaptSFL: Adaptive Split Federated Learning in
# Resource-Constrained Edge Networks", IEEE ToN, Vol. 33, No. 6, 2025
# ═══════════════════════════════════════════════════════════════════════

def _adaptsfl_per_round_latency(
    cut: int, L: int, cluster_size: int,
    cfg: SimulationConfig, rates: Dict[str, float],
) -> float:
    """Per-round split training latency for AdaptSFL (Eq. 26, single round).

    AdaptSFL architecture (two-tier):
        • Client-side (IoT):       layers [0, cut)
        • Server-side (Edge Srv):  layers [cut, L)

    Latency components from mta_sfl.the paper:
        T_split = max_i{T^F_i + T^U_{a,i}} + T^F_s + T^B_s
                  + max_i{T^D_{g,i} + T^B_i}

    Simplified for homogeneous IoT devices within a cluster.
    """
    # --- Client-side Forward Propagation (Eq. 16) ---
    # T^F_i = b · Φ^F_{c,i}(α) / f_i
    T_fp_client = cut * FLOPS_PER_BLOCK / cfg.compute.f_iot_hz

    # --- Activation Upload (Eq. 17) ---
    # T^U_{a,i} = b · φ_{a,i}(α) / r^U_i
    activation_bits = 1.0e6
    R_upload = max(rates.get("R_iot_primary", MIN_RATE_BPS), MIN_RATE_BPS)
    T_upload = min(activation_bits / R_upload, MAX_COMM_DUR_S)

    # --- Server-side FP + BP (Eq. 18-19) ---
    # Server processes concatenated batch from mta_sfl.all cluster IoTs
    server_flops = (L - cut) * FLOPS_PER_BLOCK * cluster_size
    T_fp_server = server_flops / cfg.compute.f_uav_hz
    # BP ≈ 2× FP compute (standard approximation)
    T_bp_server = 2.0 * server_flops / cfg.compute.f_uav_hz

    # --- Gradient Download (Eq. 20) ---
    # T^D_{g,i} = b · φ_{g,i}(α) / r^D_i
    # Gradient size ≈ activation size at the cut layer
    T_download = min(activation_bits / R_upload, MAX_COMM_DUR_S)

    # --- Client-side Back Propagation (Eq. 21) ---
    # T^B_i = b · Ψ_{c,i}(α) / f_i ; BP ≈ 2× FP compute
    T_bp_client = 2.0 * cut * FLOPS_PER_BLOCK / cfg.compute.f_iot_hz

    # --- Total per-round latency (Eq. 26, without MA overhead) ---
    T_split = ((T_fp_client + T_upload)
               + T_fp_server + T_bp_server
               + (T_download + T_bp_client))
    return T_split


def _adaptsfl_dinkelbach_cut(
    L: int, cluster_size: int,
    cfg: SimulationConfig, rates: Dict[str, float],
    N_total_iots: int,
    beta: float = 1.0,
    eta: float = 0.01,
    G_sq_per_layer: float = 1.0,
    epsilon: float = 0.5,
    max_iter: int = 20,
    delta: float = 1e-6,
) -> int:
    """Dinkelbach algorithm (Algorithm 2) for optimal cut layer selection.

    Minimises the convergence-latency ratio from Problem P2 (Eq. 34):

        Ψ(α) = T_split(α) / Q(α)

    where Q(α) = ε − 4β²η² · Σ_{j∈client} G²_j  is the convergence
    quality factor derived from mta_sfl.the bound in Theorem 1.

    The Dinkelbach method transforms this into iterating:
        1.  Fix λ.  Solve: min_{cut}  T(cut) − λ · Q(cut)
        2.  Update:  λ ← T(cut*) / Q(cut*)
    until |λ^{n+1} − λ^{n}| < δ.

    Parameters match the paper's notation:
        β  : L-smoothness constant of the loss (Assumption 1)
        η  : learning rate
        G² : per-layer second-moment bound (Assumption 2)
        ε  : target convergence accuracy
    """
    lam = 0.0  # Dinkelbach parameter λ^(n)
    best_cut = max(1, L // 2)

    for _n in range(max_iter):
        best_subobj = float("inf")

        for cut in range(1, L):                       # cut ∈ {1, …, L−1}
            T_cut = _adaptsfl_per_round_latency(cut, L, cluster_size, cfg, rates)

            # Convergence quality Q(cut) from Theorem 1 / Corollary 1
            client_divergence = 4.0 * beta ** 2 * eta ** 2 * cut * G_sq_per_layer
            Q_cut = max(epsilon - client_divergence, 1e-10)

            # Dinkelbach sub-problem: min T(cut) − λ · Q(cut)
            subobj = T_cut - lam * Q_cut
            if subobj < best_subobj:
                best_subobj = subobj
                best_cut = cut

        # Update λ^(n+1) = T(cut*) / Q(cut*)
        T_star = _adaptsfl_per_round_latency(best_cut, L, cluster_size, cfg, rates)
        div_star = 4.0 * beta ** 2 * eta ** 2 * best_cut * G_sq_per_layer
        Q_star = max(epsilon - div_star, 1e-10)
        lam_new = T_star / Q_star

        if abs(lam_new - lam) < delta:
            break
        lam = lam_new

    return max(1, min(best_cut, L - 1))

class SharedSimulationRunner:
    def __init__(self, baseline_name):
        self.baseline_name = baseline_name
        
    def run(self, dataset_name: str, episode: int) -> EpisodeMetrics:
        cfg = default_config()
        cfg.num_iot = NUM_IOTS
        cfg.num_uav = NUM_UAVS
        cfg.num_cloud = NUM_CLOUD
        cfg.seed = SEED
        
        rng = np.random.default_rng(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
        
        data_partitioner = DataPartitioner(
            dataset_name=dataset_name, num_clients=cfg.num_iot,
            mode="dirichlet", alpha=DIRICHLET_ALPHA, batch_size=32,
            data_root="data", seed=cfg.seed
        )
        client_loaders, test_loader = data_partitioner.partition()
        num_classes = data_partitioner.num_classes
        
        iots, uavs, clouds = initialize_entities(cfg, rng)
        all_entities = list(iots) + list(uavs) + list(clouds)
        iot_indices = [d.node_id for d in iots]
        uav_indices = [u.node_id for u in uavs]
        cloud_idx = clouds[0].node_id
        
        mobility = GaussMarkovMobility(cfg.mobility, seed=cfg.seed)
        positions, node_types = _collect_positions_and_types(iots, uavs, clouds)
        channel = ChannelState(positions, node_types, cfg, rng)
        topology = TopologyManager(cfg)
        energy_ledger = EnergyLedger(cfg.num_iot, cfg.num_uav, cfg.num_cloud, cfg)
        sync_barrier = GlobalSyncBarrier()
        tx_powers = _build_tx_powers(cfg, len(all_entities))
        
        L = SplitModel(dataset_name, num_classes).L
        round_latencies: List[float] = []
        round_energies: List[float] = []
        round_accuracies: List[float] = []
        final_accuracy = 0.0
        total_latency_accum = 0.0
        total_energy_accum = 0.0
        
        ref_model = SplitModel(dataset_name, num_classes)
        global_model_state_dict = copy.deepcopy(ref_model.state_dict())
        global_momentum = {k: torch.zeros_like(v) for k, v in global_model_state_dict.items()}
        global_compression_manager = CompressionManager(TOP_K_RATIO)
        
        for t in range(1, NUM_ROUNDS + 1):
            mobility.step(uavs, cfg.dt_s)
            positions, _ = _collect_positions_and_types(iots, uavs, clouds)
            channel.update(positions)
            
            preliminary_assoc = {u: [] for u in uav_indices}
            for iot in iots:
                if not energy_ledger.is_alive(iot.node_id):
                    continue
                best_u = max(uav_indices, key=lambda u: channel.gain_matrix[iot.node_id, u])
                preliminary_assoc[best_u].append(iot.node_id)
                
            sinr_matrix = compute_all_sinr(channel, preliminary_assoc, tx_powers, cfg)
            associations = ofdma_association(sinr_matrix, iots, uav_indices, energy_ledger, 5)
            
            active_primaries = sorted(associations.keys())
            if not active_primaries:
                if t % 10 == 0 or t == NUM_ROUNDS:
                    round_latencies.append(0.0)
                    round_energies.append(total_energy_accum)
                    round_accuracies.append(final_accuracy)
                continue
                
            sinr_matrix = compute_all_sinr(channel, associations, tx_powers, cfg)
            topology.update(sinr_matrix)
            
            cluster_configs = []
            relay_selections = {}
            split_selections = {}
            
            graph = nx.erdos_renyi_graph(n=len(uav_indices), p=0.3, seed=cfg.seed)
            uav_neighbors = {uav_indices[i]: [uav_indices[j] for j in graph.neighbors(i)] for i in range(len(uav_indices))}
            
            # 3d. Baseline logic
            for primary_idx in active_primaries:
                iot_list = associations[primary_idx]
                
                best_relay = primary_idx
                best_l = 1
                
                if self.baseline_name == "FedATC":
                    best_cost = float('inf')
                    neighbors = uav_neighbors[primary_idx] + [primary_idx]
                    for r in neighbors:
                        if not energy_ledger.is_alive(r): continue
                        for l in range(1, L + 1):
                            e, _ = estimate_pipeline_cost(primary_idx, r, l, iot_list, sinr_matrix, cfg, cloud_idx, L)
                            if e < best_cost:
                                best_cost, best_relay, best_l = e, r, l
                elif self.baseline_name == "SFLSCC":
                    best_cost = float('inf')
                    neighbors = uav_neighbors[primary_idx] + [primary_idx]
                    for r in neighbors:
                        if not energy_ledger.is_alive(r): continue
                        for l in range(1, L + 1):
                            _, lat = estimate_pipeline_cost(primary_idx, r, l, iot_list, sinr_matrix, cfg, cloud_idx, L)
                            if lat < best_cost:
                                best_cost, best_relay, best_l = lat, r, l
                elif self.baseline_name == "AdaptSFL":
                    # ── AdaptSFL (Lin et al., IEEE ToN 2025) ──────────
                    # Two-tier split: IoT [0,cut) + Edge Server (UAV) [cut,L)
                    # No relay; data flows directly IoT → UAV (edge server).
                    # Cut layer selected via Dinkelbach algorithm (Alg. 2)
                    # minimising the convergence-latency ratio Ψ(α).
                    best_relay = primary_idx  # No relay in AdaptSFL
                    cluster_size = len(iot_list)
                    rep_iot = iot_list[0]
                    rates_local = _pipeline_rates(
                        sinr_matrix, rep_iot, primary_idx, primary_idx,
                        cloud_idx, cfg, cluster_size,
                    )
                    N_total = sum(len(v) for v in associations.values())
                    best_l = _adaptsfl_dinkelbach_cut(
                        L=L, cluster_size=cluster_size, cfg=cfg,
                        rates=rates_local, N_total_iots=N_total,
                        eta=0.01,   # matches orchestrator lr
                    )

                elif self.baseline_name == "SHSFL":
                    median_res = np.median([energy_ledger.get_residual(u) for u in uav_indices if energy_ledger.is_alive(u)])
                    if energy_ledger.get_residual(primary_idx) >= median_res:
                        best_l = L
                    else:
                        best_l = 1
                    best_relay = primary_idx

                # ── Split mapping ──────────────────────────────────────
                # AdaptSFL: two-tier (IoT [0,cut), UAV [cut,L))
                # Others:   three-tier (IoT [0,1), UAV [1,l), Cloud [l,L))
                if self.baseline_name == "AdaptSFL":
                    split = (best_l, L, L)
                else:
                    split = (1, best_l, best_l)

                relay_selections[primary_idx] = best_relay
                split_selections[primary_idx] = split
                
                cluster_configs.append(ClusterConfig(
                    cluster_id=f"cluster_{primary_idx}",
                    primary_uav_id=str(primary_idx),
                    relay_uav_id=str(best_relay) if best_relay != primary_idx else None,
                    iot_ids=[str(i) for i in iot_list],
                    cut_config=split,
                ))

            topology.queue.reset()
            orchestrator = SFLOrchestrator(
                dataset_name=dataset_name, num_classes=num_classes,
                clusters=cluster_configs, k_ratio=TOP_K_RATIO,
                lr=0.01, optimizer_name="sgd", momentum=0.0, device=DEVICE,
                global_state_dict=global_model_state_dict,
                global_compression_manager=global_compression_manager
            )
            
            iot_loaders = {}
            for primary_idx, iot_list in associations.items():
                for iot_id in iot_list:
                    iot_loaders[str(iot_id)] = client_loaders[iot_id]
                    
            round_metrics = orchestrator.run_round(round_idx=t, iot_loaders=iot_loaders, test_loader=test_loader)
            
            # Extract the new aggregated global model state
            new_state_dict = orchestrator.assemble_global_model().state_dict()
            
            # Apply Server Momentum (FedAvgM)
            for k in global_model_state_dict.keys():
                if "weight" in k or "bias" in k:
                    pseudo_grad = global_model_state_dict[k] - new_state_dict[k]
                    global_momentum[k] = 0.9 * global_momentum[k] + pseudo_grad
                    global_model_state_dict[k] -= global_momentum[k]
                else:
                    global_model_state_dict[k] = new_state_dict[k]
            
            test_loss, test_acc, per_class_acc = orchestrator.global_evaluate(test_loader)
            
            # Energy calculation
            task_durations = {}
            rf_durations = {}
            comp_flops = {}
            node_types_map = {}
            
            for primary_idx, iot_list in associations.items():
                relay_idx = relay_selections[primary_idx]
                p1, p2, p3 = split_selections[primary_idx]
                cluster_size = len(iot_list)
                rates = _pipeline_rates(sinr_matrix, iot_list[0], primary_idx, relay_idx, cloud_idx, cfg, cluster_size)
                
                flops_iot = p1 * FLOPS_PER_BLOCK * cluster_size
                flops_prim = (p2 - p1) * FLOPS_PER_BLOCK * cluster_size
                flops_relay = (p3 - p2) * FLOPS_PER_BLOCK * cluster_size
                
                t_comp_iot = flops_iot / cfg.compute.f_iot_hz
                t_comp_prim = flops_prim / cfg.compute.f_uav_hz
                t_comp_relay = flops_relay / cfg.compute.f_uav_hz
                
                activation_bits = 1.0e6
                t_comm_ip = min(activation_bits / max(rates["R_iot_primary"], MIN_RATE_BPS), MAX_COMM_DUR_S)
                t_comm_pr = 0.0 if relay_idx == primary_idx else min(activation_bits / max(rates["R_primary_relay"], MIN_RATE_BPS), MAX_COMM_DUR_S)
                t_comm_rc = min(activation_bits / max(rates["R_relay_cloud"], MIN_RATE_BPS), MAX_COMM_DUR_S) if p3 < L else 0.0
                
                for iot_id in iot_list:
                    task_durations[iot_id] = t_comp_iot / cluster_size + t_comm_ip
                    rf_durations[iot_id] = (cfg.power.P_tx_iot, t_comm_ip, cfg.power.P_rx_iot, t_comm_ip)
                    comp_flops[iot_id] = p1 * FLOPS_PER_BLOCK
                    node_types_map[iot_id] = "iot"
                    
                task_durations[primary_idx] = task_durations.get(primary_idx, 0.0) + t_comp_prim + t_comm_ip + t_comm_pr
                prev_rf = rf_durations.get(primary_idx, (cfg.power.P_tx_uav, 0.0, cfg.power.P_rx_uav, 0.0))
                rf_durations[primary_idx] = (cfg.power.P_tx_uav, prev_rf[1] + t_comm_pr, cfg.power.P_rx_uav, prev_rf[3] + t_comm_ip)
                comp_flops[primary_idx] = comp_flops.get(primary_idx, 0.0) + (p2 - p1) * FLOPS_PER_BLOCK * cluster_size
                node_types_map[primary_idx] = "uav"
                
                topology.queue.enqueue(primary_idx, activation_bits * cluster_size)
                
                if relay_idx != primary_idx:
                    task_durations[relay_idx] = task_durations.get(relay_idx, 0.0) + t_comp_relay + t_comm_pr + t_comm_rc
                    prev_rf = rf_durations.get(relay_idx, (cfg.power.P_tx_uav, 0.0, cfg.power.P_rx_uav, 0.0))
                    rf_durations[relay_idx] = (cfg.power.P_tx_uav, prev_rf[1] + t_comm_rc, cfg.power.P_rx_uav, prev_rf[3] + t_comm_pr)
                    comp_flops[relay_idx] = comp_flops.get(relay_idx, 0.0) + (p3 - p2) * FLOPS_PER_BLOCK * cluster_size
                    node_types_map[relay_idx] = "uav"
                    topology.queue.enqueue(relay_idx, activation_bits * cluster_size)
                else:
                    task_durations[primary_idx] += t_comp_relay + t_comm_rc
                    prev_rf = rf_durations[primary_idx]
                    rf_durations[primary_idx] = (cfg.power.P_tx_uav, prev_rf[1] + t_comm_rc, cfg.power.P_rx_uav, prev_rf[3])
                    comp_flops[primary_idx] += (p3 - p2) * FLOPS_PER_BLOCK * cluster_size
                    
            task_durations[cloud_idx] = sum(
                (L - split_selections[p][2]) * FLOPS_PER_BLOCK * len(associations[p]) / cfg.compute.f_cloud_hz
                for p in active_primaries
            )
            rf_durations[cloud_idx] = (cfg.power.P_tx_cloud, 0.0, cfg.power.P_rx_cloud, 0.0)
            comp_flops[cloud_idx] = sum(
                (L - split_selections[p][2]) * FLOPS_PER_BLOCK * len(associations[p])
                for p in active_primaries
            )
            node_types_map[cloud_idx] = "cloud"
            
            for u_idx in uav_indices:
                if u_idx not in node_types_map and energy_ledger.is_alive(u_idx):
                    task_durations[u_idx] = 0.0
                    rf_durations[u_idx] = (0.0, 0.0, 0.0, 0.0)
                    comp_flops[u_idx] = 0.0
                    node_types_map[u_idx] = "uav"
                    
            T_global = sync_barrier.compute_T_global(task_durations)
            round_energies_dict = energy_ledger.apply_global_round(
                T_global=T_global, task_durations=task_durations,
                rf_durations=rf_durations, comp_flops=comp_flops,
                node_types=node_types_map, cfg=cfg
            )
            
            total_energy = sum(round_energies_dict.values())
            
            total_latency_accum += T_global
            total_energy_accum += total_energy
            final_accuracy = test_acc
            
            if t % 10 == 0 or t == NUM_ROUNDS:
                round_latencies.append(T_global)
                round_energies.append(total_energy_accum)
                round_accuracies.append(test_acc)
        
        # Build final EpisodeMetrics
        drones_depleted = sum(1 for u in uav_indices if not energy_ledger.is_alive(u))
        ep = EpisodeMetrics(
            episode=episode,
            global_accuracy=final_accuracy,
            avg_latency=total_latency_accum / NUM_ROUNDS if NUM_ROUNDS > 0 else 0.0,
            total_energy=total_energy_accum,
            drones_depleted=drones_depleted,
            round_accuracies=round_accuracies,
            round_latencies=round_latencies,
            round_energies=round_energies,
            extra={"baseline": self.baseline_name, "dataset": dataset_name},
        )
        return ep


class FedATCBaseline:
    def run_episode(self, dataset_name: str, episode: int) -> EpisodeMetrics:
        runner = SharedSimulationRunner("FedATC")
        return runner.run(dataset_name, episode)

class SFLSCCBaseline:
    def run_episode(self, dataset_name: str, episode: int) -> EpisodeMetrics:
        runner = SharedSimulationRunner("SFLSCC")
        return runner.run(dataset_name, episode)

class AdaptSFLBaseline:
    def run_episode(self, dataset_name: str, episode: int) -> EpisodeMetrics:
        runner = SharedSimulationRunner("AdaptSFL")
        return runner.run(dataset_name, episode)

class SHSFLBaseline:
    def run_episode(self, dataset_name: str, episode: int) -> EpisodeMetrics:
        runner = SharedSimulationRunner("SHSFL")
        return runner.run(dataset_name, episode)
