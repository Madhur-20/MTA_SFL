# mta_sfl/config – Dataclass-based simulation configuration
# Re-exports for convenient access
from .system_config import (
    PhysicalConstants,
    ChannelConfig,
    PowerConfig,
    BatteryConfig,
    ComputeConfig,
    MobilityConfig,
    SimulationConfig,
    default_config,
)

__all__ = [
    "PhysicalConstants",
    "ChannelConfig",
    "PowerConfig",
    "BatteryConfig",
    "ComputeConfig",
    "MobilityConfig",
    "SimulationConfig",
    "default_config",
]
