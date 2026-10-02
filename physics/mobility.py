"""MTA-SFL 3D Spatial Mobility, Relative Kinematics & Predicted Route Lifetime.

Tracks 3-D coordinates and velocity vectors for every entity in the simulator:
IoT devices (static, z = 0), UAV swarm (bounded altitude corridor), and Cloud
gateway base station(s).

Key functionality
-----------------
* Gauss-Markov mobility model with momentum, boundary reflection and velocity
  clipping.
* Predicted route lifetime L_life(m, n, t) via relative-kinematic quadratic
  projection (see §III-A of the MTA-SFL paper).

References
----------
  Eq. (Route Lifetime):
    Δt = [ -(r·v) + sqrt( (r·v)² − ‖v‖²·(‖r‖² − R²) ) ] / ‖v‖²
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from mta_sfl.config.system_config import MobilityConfig, SimulationConfig


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Entity3D — base representation
# ──────────────────────────────────────────────────────────────────────────────
class Entity3D:
    """Base class for any physical entity with 3-D position and velocity.

    Attributes:
        node_id: Unique integer identifier.
        node_type: One of ``'iot'``, ``'uav'``, ``'cloud'``.
        position: 3-D coordinate array ``[x, y, z]`` (metres).
        velocity: 3-D velocity vector ``[vx, vy, vz]`` (m / s).
    """

    __slots__ = ("node_id", "node_type", "position", "velocity")

    def __init__(
        self,
        node_id: int,
        node_type: str,
        position: np.ndarray,
        velocity: np.ndarray,
    ) -> None:
        self.node_id: int = node_id
        self.node_type: str = node_type
        self.position: np.ndarray = np.asarray(position, dtype=np.float64)
        self.velocity: np.ndarray = np.asarray(velocity, dtype=np.float64)

    # Convenience aliases used by legacy callers ---
    @property
    def pos(self) -> np.ndarray:
        return self.position

    @pos.setter
    def pos(self, value: np.ndarray) -> None:
        self.position = np.asarray(value, dtype=np.float64)

    @property
    def vel(self) -> np.ndarray:
        return self.velocity

    @vel.setter
    def vel(self, value: np.ndarray) -> None:
        self.velocity = np.asarray(value, dtype=np.float64)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(id={self.node_id}, "
            f"pos={self.position}, vel={self.velocity})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Concrete Entity Types
# ──────────────────────────────────────────────────────────────────────────────
class IoTDevice(Entity3D):
    """Static ground-level IoT device at *z = 0*.

    Args:
        node_id: Unique identifier.
        x: Ground-plane x coordinate (m).
        y: Ground-plane y coordinate (m).
    """

    def __init__(self, node_id: int, x: float, y: float) -> None:
        super().__init__(
            node_id=node_id,
            node_type="iot",
            position=np.array([x, y, 0.0]),
            velocity=np.zeros(3),
        )


class UAVNode(Entity3D):
    """Mobile UAV node bounded within an altitude corridor.

    Args:
        node_id: Unique identifier.
        x, y, z: Initial 3-D position (m).
        vx, vy, vz: Initial velocity components (m / s).
    """

    def __init__(
        self,
        node_id: int,
        x: float,
        y: float,
        z: float,
        vx: float = 0.0,
        vy: float = 0.0,
        vz: float = 0.0,
    ) -> None:
        super().__init__(
            node_id=node_id,
            node_type="uav",
            position=np.array([x, y, z]),
            velocity=np.array([vx, vy, vz]),
        )


class CloudGateway(Entity3D):
    """Fixed cloud-gateway base station.

    Args:
        node_id: Unique identifier.
        x, y, z: Position (m).
    """

    def __init__(self, node_id: int, x: float, y: float, z: float) -> None:
        super().__init__(
            node_id=node_id,
            node_type="cloud",
            position=np.array([x, y, z]),
            velocity=np.zeros(3),
        )


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Helper Geometry Functions
# ──────────────────────────────────────────────────────────────────────────────
def euclidean_distance(a: Entity3D, b: Entity3D) -> float:
    """Euclidean distance between two entities (m)."""
    return float(np.linalg.norm(a.position - b.position))


def relative_velocity(a: Entity3D, b: Entity3D) -> np.ndarray:
    """Relative velocity vector v_a − v_b."""
    return a.velocity - b.velocity


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Gauss-Markov Mobility Model
# ──────────────────────────────────────────────────────────────────────────────
class GaussMarkovMobility:
    """Gauss-Markov 3-D mobility model for UAV swarm motion.

    Velocity update per axis (Eq. from Camp et al. 2002):
        v(t+1) = α · v(t)  +  (1 − α) · v̄  +  σ · √(1 − α²) · N(0, 1)
    with v̄ = 0 (hovering mean), α = ``gauss_markov_alpha``,
    σ = ``gauss_markov_sigma_v``.

    Positions are updated via Euler integration and reflected off the arena
    boundaries to keep UAVs inside the operational volume.

    Args:
        config: MobilityConfig instance.
        seed: Explicit RNG seed for reproducibility.
    """

    def __init__(self, config: MobilityConfig, seed: int = 42) -> None:
        self.config: MobilityConfig = config
        self.rng: np.random.Generator = np.random.default_rng(seed)
        self._alpha: float = config.gauss_markov_alpha
        self._sigma: float = config.gauss_markov_sigma_v
        self._v_max: float = config.v_max_uav_mps
        self._bounds_lo: np.ndarray = np.array(
            [0.0, 0.0, config.z_min_m]
        )
        self._bounds_hi: np.ndarray = np.array(
            [config.arena_x_m, config.arena_y_m, config.z_max_m]
        )

    # ------------------------------------------------------------------ step
    def step(self, uavs: List[UAVNode], dt: float) -> None:
        """Advance positions and velocities of *all* UAVs by ``dt`` seconds.

        Updates are performed **in-place**.

        Args:
            uavs: List of UAV nodes to update.
            dt: Time-step duration (s).
        """
        sqrt_term = math.sqrt(max(1.0 - self._alpha ** 2, 0.0))

        for uav in uavs:
            # --- velocity update (per-axis) ---
            noise = self.rng.normal(0.0, 1.0, size=3)
            uav.velocity = (
                self._alpha * uav.velocity
                + self._sigma * sqrt_term * noise
            )

            # clip to maximum speed
            speed = float(np.linalg.norm(uav.velocity))
            if speed > self._v_max:
                uav.velocity *= self._v_max / speed

            # --- position update (Euler) ---
            uav.position = uav.position + uav.velocity * dt

            # --- boundary reflection ---
            for axis in range(3):
                lo, hi = self._bounds_lo[axis], self._bounds_hi[axis]
                if uav.position[axis] < lo:
                    uav.position[axis] = 2.0 * lo - uav.position[axis]
                    uav.velocity[axis] = abs(uav.velocity[axis])
                elif uav.position[axis] > hi:
                    uav.position[axis] = 2.0 * hi - uav.position[axis]
                    uav.velocity[axis] = -abs(uav.velocity[axis])

            # safety clamp (after reflection the position should already
            # be inside, but guard against extreme-dt overshoot)
            uav.position = np.clip(uav.position, self._bounds_lo, self._bounds_hi)


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Predicted Route Lifetime  (L_life)
# ──────────────────────────────────────────────────────────────────────────────
def compute_route_lifetime(
    entity_a: Entity3D,
    entity_b: Entity3D,
    R_comm: float,
) -> float:
    """Predicted remaining link duration before two entities leave range.

    Solves the quadratic ‖r + v·t‖² = R² for the positive root:

        ‖v‖²·t²  +  2(r·v)·t  +  (‖r‖² − R²) = 0

        Δt = [ −(r·v) + √{ (r·v)² − ‖v‖²·(‖r‖² − R²) } ] / ‖v‖²

    Edge-case handling:
    * Already out of range (d ≥ R_comm) → 0.0
    * Quasi-static (‖v_rel‖ < ε)       → +∞  (link persists indefinitely)
    * Negative discriminant             → +∞  (nodes will never leave range)
    * Negative Δt (only approaching)    → max(Δt, 0.0)

    Args:
        entity_a: First entity.
        entity_b: Second entity.
        R_comm: Communication range threshold (m).

    Returns:
        Predicted link lifetime in seconds.  ``float('inf')`` if the link
        is expected to persist indefinitely under current kinematics.
    """
    r: np.ndarray = entity_a.position - entity_b.position
    v: np.ndarray = entity_a.velocity - entity_b.velocity

    d_sq: float = float(np.dot(r, r))
    v_sq: float = float(np.dot(v, v))
    R_sq: float = R_comm * R_comm

    # Already outside communication range.
    if d_sq >= R_sq:
        return 0.0

    # Quasi-static: relative speed effectively zero.
    _EPS_V_SQ: float = 1.0e-12
    if v_sq < _EPS_V_SQ:
        return float("inf")

    r_dot_v: float = float(np.dot(r, v))
    discriminant: float = r_dot_v * r_dot_v - v_sq * (d_sq - R_sq)

    # Negative discriminant ⇒ trajectory never exits the sphere.
    if discriminant < 0.0:
        return float("inf")

    delta_t: float = (-r_dot_v + math.sqrt(discriminant)) / v_sq
    return max(delta_t, 0.0)


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Entity Factory
# ──────────────────────────────────────────────────────────────────────────────
def initialize_entities(
    cfg: SimulationConfig,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[IoTDevice], List[UAVNode], List[CloudGateway]]:
    """Create all simulation entities with randomised initial positions.

    IoT devices are scattered uniformly on the ground plane.  UAVs are placed
    uniformly within the 3-D operational volume with initial velocities drawn
    uniformly and clipped to ``v_max``.  A single Cloud gateway is placed at
    the arena centre above the altitude ceiling.

    Args:
        cfg: Full simulation configuration.
        rng: Optional pre-seeded NumPy Generator.  If *None* a new one is
             created from ``cfg.seed``.

    Returns:
        Tuple of ``(iot_devices, uavs, cloud_gateways)``.
    """
    if rng is None:
        rng = np.random.default_rng(cfg.seed)

    mob = cfg.mobility

    # --- IoT devices (ground, z = 0) ---
    iots: List[IoTDevice] = []
    for i in range(cfg.num_iot):
        x = rng.uniform(0.0, mob.arena_x_m)
        y = rng.uniform(0.0, mob.arena_y_m)
        iots.append(IoTDevice(node_id=i, x=x, y=y))

    # --- UAV nodes ---
    uavs: List[UAVNode] = []
    offset = cfg.num_iot
    for j in range(cfg.num_uav):
        x = rng.uniform(0.0, mob.arena_x_m)
        y = rng.uniform(0.0, mob.arena_y_m)
        z = rng.uniform(mob.z_min_m, mob.z_max_m)

        vel_raw = rng.uniform(-mob.v_max_uav_mps, mob.v_max_uav_mps, size=3)
        speed = float(np.linalg.norm(vel_raw))
        if speed > mob.v_max_uav_mps:
            vel_raw = vel_raw * (mob.v_max_uav_mps / speed)

        uavs.append(
            UAVNode(
                node_id=offset + j,
                x=x, y=y, z=z,
                vx=float(vel_raw[0]),
                vy=float(vel_raw[1]),
                vz=float(vel_raw[2]),
            )
        )

    # --- Cloud gateway(s) ---
    clouds: List[CloudGateway] = []
    cloud_offset = cfg.num_iot + cfg.num_uav
    for k in range(cfg.num_cloud):
        clouds.append(
            CloudGateway(
                node_id=cloud_offset + k,
                x=mob.arena_x_m / 2.0,
                y=mob.arena_y_m / 2.0,
                z=mob.z_max_m + 100.0,  # above altitude ceiling
            )
        )

    return iots, uavs, clouds


# ──────────────────────────────────────────────────────────────────────────────
# Self-test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running mobility.py self-tests …")

    cfg = SimulationConfig(num_iot=10, num_uav=4, num_cloud=1, seed=123)
    mob = cfg.mobility

    # ----- 1. Closing UAVs ⇒ finite positive L_life -----
    uav_a = UAVNode(0, 100.0, 100.0, 100.0, vx=10.0, vy=0.0, vz=0.0)
    uav_b = UAVNode(1, 200.0, 100.0, 100.0, vx=-5.0, vy=0.0, vz=0.0)
    d_ab = euclidean_distance(uav_a, uav_b)
    assert d_ab == 100.0, f"Distance mismatch: {d_ab}"

    life = compute_route_lifetime(uav_a, uav_b, R_comm=300.0)
    assert 0.0 < life < float("inf"), f"Expected finite positive, got {life}"
    print(f"  [OK] Closing UAVs: d={d_ab:.0f} m, L_life={life:.2f} s")

    # ----- 2. Static entities within range ⇒ L_life = ∞ -----
    static_a = IoTDevice(10, 50.0, 50.0)
    static_b = IoTDevice(11, 100.0, 50.0)
    life_static = compute_route_lifetime(static_a, static_b, R_comm=300.0)
    assert life_static == float("inf"), f"Expected inf, got {life_static}"
    print(f"  [OK] Static in-range: L_life = inf")

    # ----- 3. Already out of range ⇒ L_life = 0 -----
    far_a = UAVNode(20, 0.0, 0.0, 100.0)
    far_b = UAVNode(21, 900.0, 900.0, 100.0)
    life_oor = compute_route_lifetime(far_a, far_b, R_comm=300.0)
    assert life_oor == 0.0, f"Expected 0.0, got {life_oor}"
    print(f"  [OK] Out-of-range: L_life = 0.0")

    # ----- 4. Diverging UAVs ⇒ finite L_life -----
    div_a = UAVNode(30, 100.0, 100.0, 100.0, vx=10.0, vy=0.0, vz=0.0)
    div_b = UAVNode(31, 150.0, 100.0, 100.0, vx=-10.0, vy=0.0, vz=0.0)
    life_div = compute_route_lifetime(div_a, div_b, R_comm=300.0)
    assert 0.0 < life_div < float("inf"), f"Expected finite positive, got {life_div}"
    print(f"  [OK] Diverging UAVs: L_life = {life_div:.2f} s")

    # ----- 5. Gauss-Markov keeps UAVs within bounds -----
    gm = GaussMarkovMobility(mob, seed=99)
    test_uavs = [UAVNode(50 + i, 500.0, 500.0, 100.0) for i in range(4)]

    for step_i in range(200):
        gm.step(test_uavs, dt=1.0)
        for u in test_uavs:
            assert 0.0 <= u.position[0] <= mob.arena_x_m, (
                f"Step {step_i}: x={u.position[0]} OOB"
            )
            assert 0.0 <= u.position[1] <= mob.arena_y_m, (
                f"Step {step_i}: y={u.position[1]} OOB"
            )
            assert mob.z_min_m <= u.position[2] <= mob.z_max_m, (
                f"Step {step_i}: z={u.position[2]} OOB"
            )
            speed = float(np.linalg.norm(u.velocity))
            assert speed <= mob.v_max_uav_mps + 1e-9, (
                f"Step {step_i}: speed={speed} exceeds v_max"
            )
    print(f"  [OK] Gauss-Markov: 200 steps, all UAVs in bounds, speed ≤ v_max")

    # ----- 6. Entity factory -----
    iots, uavs, clouds = initialize_entities(cfg)
    assert len(iots) == cfg.num_iot
    assert len(uavs) == cfg.num_uav
    assert len(clouds) == cfg.num_cloud
    for iot in iots:
        assert iot.position[2] == 0.0
    for uav in uavs:
        assert mob.z_min_m <= uav.position[2] <= mob.z_max_m
    print(f"  [OK] Entity factory: {len(iots)} IoT, {len(uavs)} UAV, {len(clouds)} Cloud")

    print("[PASS] mobility.py — All assertions passed.")
