# mta_sfl/physics – Physical-layer simulation models
from .mobility import (
    Entity3D,
    IoTDevice,
    UAVNode,
    CloudGateway,
    GaussMarkovMobility,
    compute_route_lifetime,
)
from .channel import (
    path_loss_free_space_dB,
    channel_gain,
    ofdma_subchannel_bandwidth,
    compute_sinr,
    shannon_rate,
    interference_impact_telemetry,
    ChannelState,
)
from .topology import (
    build_adjacency_matrix,
    local_clustering_coefficient,
    QueueTracker,
    TopologyManager,
)
from .energy import (
    EnergyLedger,
    compute_propulsion_energy,
    compute_circuit_energy_uav,
    compute_circuit_energy_iot,
    compute_rf_energy,
    compute_computation_energy,
    GlobalSyncBarrier,
)

__all__ = [
    # mobility
    "Entity3D", "IoTDevice", "UAVNode", "CloudGateway",
    "GaussMarkovMobility", "compute_route_lifetime",
    # channel
    "path_loss_free_space_dB", "channel_gain", "ofdma_subchannel_bandwidth",
    "compute_sinr", "shannon_rate", "interference_impact_telemetry", "ChannelState",
    # topology
    "build_adjacency_matrix", "local_clustering_coefficient",
    "QueueTracker", "TopologyManager",
    # energy
    "EnergyLedger", "compute_propulsion_energy", "compute_circuit_energy_uav",
    "compute_circuit_energy_iot", "compute_rf_energy",
    "compute_computation_energy", "GlobalSyncBarrier",
]
