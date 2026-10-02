"""MTA-SFL System Configuration — Dataclass-based physical constants, RF specs,
battery parameters, and compute profiles.

Every numeric default is derived from the MTA-SFL paper or from standard UAV /
IoT-link budgets cited therein.  All SI units unless noted otherwise.

Reference carrier frequencies:
    f_c = 2.4 GHz  (IoT-to-UAV uplink, default)
    f_c = 5.8 GHz  (UAV-to-UAV backhaul / Cloud link, optional)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Fundamental Physical Constants
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PhysicalConstants:
    """Immutable physical constants used throughout the channel and energy models.

    Attributes:
        c: Speed of light in vacuum (m/s).
        boltzmann_k: Boltzmann constant (J/K).
        ambient_temp: Reference ambient temperature (K).  290 K is the standard
            noise-temperature assumption in the ITU-R thermal noise model.
        N0_dBm_per_Hz: Thermal noise power spectral density (dBm/Hz).
            Default -174 dBm/Hz corresponds to kT at 290 K.
        N0_W_per_Hz: Same quantity in linear Watts/Hz (derived).
    """
    c: float = 299_792_458.0                        # m/s
    boltzmann_k: float = 1.380649e-23               # J/K
    ambient_temp: float = 290.0                      # K
    N0_dBm_per_Hz: float = -174.0                    # dBm/Hz

    @property
    def N0_W_per_Hz(self) -> float:
        """Thermal noise PSD in linear Watts/Hz."""
        return 10.0 ** ((self.N0_dBm_per_Hz - 30.0) / 10.0)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Channel & RF Configuration
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class ChannelConfig:
    """Radio-frequency and propagation parameters.

    Bandwidth partitions follow the MTA-SFL multi-tier link hierarchy:
        IoT → Primary-UAV  (uplink)
        Primary-UAV → IoT  (downlink)
        UAV ↔ UAV           (backhaul)
        UAV ↔ Cloud          (directional)

    Attributes:
        f_c_iot_uav_hz: Carrier frequency for IoT-to-UAV links (Hz).
        f_c_uav_uav_hz: Carrier frequency for UAV-to-UAV backhaul (Hz).
        f_c_uav_cloud_hz: Carrier frequency for UAV-to-Cloud links (Hz).
        W_UL_hz: Total uplink bandwidth per cluster, IoT → Primary-UAV (Hz).
        W_DL_hz: Total downlink bandwidth per cluster, Primary-UAV → IoT (Hz).
        W_backhaul_hz: Bandwidth for UAV ↔ UAV backhaul links (Hz).
        W_cloud_hz: Bandwidth for UAV ↔ Cloud directional links (Hz).
        alpha_los: Path-loss exponent for line-of-sight (LoS) links.
        alpha_nlos: Path-loss exponent for non-line-of-sight (NLoS) links.
        shadow_sigma_dB: Log-normal shadowing standard deviation (dB).
        sinr_threshold_dB: Minimum SINR for link feasibility γ_th (dB).
        sinr_threshold_linear: γ_th in linear scale (derived).
    """
    # Carrier frequencies
    f_c_iot_uav_hz: float = 2.4e9       # 2.4 GHz
    f_c_uav_uav_hz: float = 5.8e9       # 5.8 GHz
    f_c_uav_cloud_hz: float = 5.8e9     # 5.8 GHz

    # Bandwidth partitions (Hz)
    W_UL_hz: float = 20.0e6             # 20 MHz IoT-to-UAV uplink
    W_DL_hz: float = 20.0e6             # 20 MHz UAV-to-IoT downlink
    W_backhaul_hz: float = 40.0e6       # 40 MHz UAV-to-UAV backhaul
    W_cloud_hz: float = 40.0e6          # 40 MHz UAV-to-Cloud

    # Path-loss exponents
    alpha_los: float = 2.0               # free-space / LoS
    alpha_nlos: float = 3.5              # obstructed / NLoS

    # Log-normal shadowing
    shadow_sigma_dB: float = 4.0         # σ = 4 dB

    # Decoding / connectivity threshold
    sinr_threshold_dB: float = 3.0       # γ_th  (≈ 3 dB typical)

    @property
    def sinr_threshold_linear(self) -> float:
        """γ_th in linear scale."""
        return 10.0 ** (self.sinr_threshold_dB / 10.0)


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Power Budget Configuration
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class PowerConfig:
    """Transmit, receive, circuit, and propulsion power parameters (Watts).

    All values follow the ranges specified in the MTA-SFL paper §III-D and
    standard UAV power profiles from DJI Matrice-class references.

    Attributes:
        P_tx_iot: IoT device transmit power (W).  Paper: 0.1 W (20 dBm).
        P_tx_uav: UAV transmit power (W).  Paper: 0.5 W (27 dBm).
        P_tx_cloud: Cloud gateway transmit power (W).  Paper: 2.0 W (33 dBm).
        P_rx_iot: IoT device receiver chain power (W).
        P_rx_uav: UAV receiver chain power (W).
        P_rx_cloud: Cloud receiver chain power (W).
        P_circuit_iot_active: Active-state hardware circuit power for IoT (W).
        P_circuit_uav: Active-state hardware circuit power for UAV (W).
        P_circuit_cloud: Active-state hardware circuit power for Cloud (W).
        P_sleep_iot: Sleep / idle power for IoT devices (W).
        P_flight_hover: UAV hover propulsion power (W).  Eq. 26.
        P_flight_cruise: UAV cruise propulsion power (W).  Typically > hover.
    """
    # Active RF Transmission powers
    P_tx_iot: float = 0.1                # 0.1 W  (100 mW, 20 dBm)
    P_tx_uav: float = 0.5               # 0.5 W  (500 mW, 27 dBm)
    P_tx_cloud: float = 2.0             # 2.0 W  (33 dBm)

    # Active RF Reception powers
    P_rx_iot: float = 0.1               # 100 mW
    P_rx_uav: float = 0.3               # 300 mW
    P_rx_cloud: float = 0.3             # 300 mW

    # Hardware baseline circuit power (active state)
    P_circuit_iot_active: float = 2.0    # 2 W active
    P_circuit_uav: float = 5.0          # 5 W active
    P_circuit_cloud: float = 10.0       # 10 W active (grid-powered, tracked)

    # Sleep / idle power
    P_sleep_iot: float = 0.05           # 50 mW
    P_sleep_uav: float = 1.0            # 1 W standby avionics
    P_sleep_cloud: float = 5.0          # 5 W standby

    # UAV flight propulsion (rotor mechanical power)
    P_flight_hover: float = 80.0        # 80 W hovering (Eq. 26)
    P_flight_cruise: float = 100.0      # 100 W cruising


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Battery & Energy Budget
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class BatteryConfig:
    """Battery capacity and safety threshold parameters.

    Attributes:
        E_init_iot_J: Initial battery energy for IoT devices (Joules).
            Default 50 kJ ≈ 3.7 V × 3750 mAh (typical AA-pack / Li-Po).
        E_init_uav_J: Initial battery energy for UAVs (Joules).
            Default 266.4 kJ ≈ 22.2 V × 3330 mAh × 4-cell (DJI-class).
        E_init_cloud_J: Cloud gateway energy reservoir (Joules).
            Set very large (grid-backed); tracked for accounting.
        E_min_iot_J: Absolute minimum safe residual for IoT (Joules).
        E_min_uav_J: Absolute minimum safe residual for UAV (Joules).
            Below this value the UAV must return to base.
        gamma_safe: Fractional safety threshold (0–1).  An entity is
            considered energy-critical when E_res / E_init < gamma_safe.
    """
    E_init_iot_J: float = 50_000.0       # 50 kJ
    E_init_uav_J: float = 266_400.0      # 266.4 kJ  (≈ 22.2 V × 3.33 Ah × 4)
    E_init_cloud_J: float = 1.0e9        # effectively grid-powered

    E_min_iot_J: float = 5_000.0         # 5 kJ
    E_min_uav_J: float = 26_640.0        # 10 % of init
    E_min_cloud_J: float = 0.0           # grid-backed

    gamma_safe: float = 0.15             # 15 % fraction


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Compute Profile
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class ComputeConfig:
    """CPU / GPU compute parameters for energy and latency modelling.

    Attributes:
        kappa: Effective switched capacitance coefficient (F).
            Standard CMOS value ~10⁻²⁷ for modern SoCs.
        f_iot_hz: Base CPU clock frequency for IoT devices (Hz).
        f_uav_hz: CPU/GPU frequency for edge UAVs (Hz).
        f_cloud_hz: Server-class clock frequency at Cloud (Hz).
        flops_per_sample: FLOPs required per data sample for one forward+backward pass.
            Typical small-model FL workload; calibrated per-experiment.
    """
    kappa: float = 1.0e-27               # switched capacitance coefficient
    f_iot_hz: float = 500.0e6            # 500 MHz (Cortex-M / RISC-V)
    f_uav_hz: float = 2.0e9             # 2 GHz   (Jetson-class)
    f_cloud_hz: float = 4.0e9           # 4 GHz   (server-class)
    flops_per_sample: float = 1.0e6      # 1 MFLOP/sample (model-dependent)


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Mobility / Arena Configuration
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class MobilityConfig:
    """3D operational arena and mobility model parameters.

    Attributes:
        arena_x_m: Arena extent along x-axis (metres).
        arena_y_m: Arena extent along y-axis (metres).
        z_min_m: Minimum UAV altitude (metres AGL).
        z_max_m: Maximum UAV altitude (metres AGL).
        v_max_uav_mps: Maximum UAV speed (m/s).
        v_max_iot_mps: Maximum IoT device speed (m/s).  0 → static.
        gauss_markov_alpha: Memory / correlation parameter α ∈ [0, 1] for the
            Gauss-Markov mobility model.  0 = fully random, 1 = constant.
        gauss_markov_sigma_v: Velocity perturbation std-dev (m/s) per axis.
        R_comm_m: Default communication range (metres).  Used as an initial
            estimate; the true range is derived from SINR ≥ γ_th via the
            channel model.
    """
    arena_x_m: float = 1000.0           # 1 km
    arena_y_m: float = 1000.0           # 1 km
    z_min_m: float = 50.0               # 50 m  AGL
    z_max_m: float = 150.0              # 150 m AGL

    v_max_uav_mps: float = 20.0         # 20 m/s  (~72 km/h)
    v_max_iot_mps: float = 0.0          # static ground devices

    gauss_markov_alpha: float = 0.8      # strong momentum
    gauss_markov_sigma_v: float = 2.0    # m/s per axis

    R_comm_m: float = 300.0              # communication range (m)


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Top-level Simulation Configuration
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class SimulationConfig:
    """Aggregated configuration for one simulation run.

    Collects all sub-configurations and adds simulation-level knobs such as
    the number of entities and the global random seed.

    Attributes:
        num_iot: Number of IoT devices.
        num_uav: Number of UAVs in the swarm.
        num_cloud: Number of Cloud gateway base stations.
        dt_s: Discrete-time simulation step size (seconds).
        seed: Master RNG seed for reproducibility.
    """
    num_iot: int = 40
    num_uav: int = 8
    num_cloud: int = 1
    dt_s: float = 1.0                    # 1-second time step

    seed: int = 42

    physics: PhysicalConstants = field(default_factory=PhysicalConstants)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    power: PowerConfig = field(default_factory=PowerConfig)
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    mobility: MobilityConfig = field(default_factory=MobilityConfig)


def default_config() -> SimulationConfig:
    """Return a default SimulationConfig instance with paper-standard values."""
    return SimulationConfig()


# ──────────────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = default_config()

    # ---- Physical constants sanity ----
    assert abs(cfg.physics.c - 299_792_458.0) < 1.0, "Speed of light mismatch"
    assert abs(cfg.physics.N0_dBm_per_Hz - (-174.0)) < 0.01, "N0 mismatch"
    # N0 linear ≈ 3.98e-21 W/Hz
    assert 3.9e-21 < cfg.physics.N0_W_per_Hz < 4.1e-21, (
        f"N0 linear conversion failed: {cfg.physics.N0_W_per_Hz:.4e}"
    )

    # ---- Channel config sanity ----
    assert cfg.channel.W_UL_hz == 20.0e6
    expected_gamma_lin = 10.0 ** (3.0 / 10.0)  # ≈ 1.9953
    assert abs(cfg.channel.sinr_threshold_linear - expected_gamma_lin) < 0.01

    # ---- Power budget sanity ----
    assert cfg.power.P_tx_iot == 0.1
    assert cfg.power.P_tx_uav == 0.5
    assert cfg.power.P_tx_cloud == 2.0
    assert 60.0 <= cfg.power.P_flight_hover <= 100.0, "Hover power out of spec"

    # ---- Battery sanity ----
    assert cfg.battery.E_init_uav_J > cfg.battery.E_min_uav_J
    assert 0.0 < cfg.battery.gamma_safe < 1.0

    # ---- Compute sanity ----
    assert cfg.compute.kappa == 1.0e-27
    assert cfg.compute.f_uav_hz > cfg.compute.f_iot_hz

    # ---- Mobility sanity ----
    assert cfg.mobility.z_min_m < cfg.mobility.z_max_m

    print("[PASS] system_config.py — All assertions passed.")
    print(f"       N0 = {cfg.physics.N0_W_per_Hz:.4e} W/Hz")
    print(f"       γ_th = {cfg.channel.sinr_threshold_linear:.4f} (linear)")
    print(f"       E_init_uav = {cfg.battery.E_init_uav_J:.0f} J")