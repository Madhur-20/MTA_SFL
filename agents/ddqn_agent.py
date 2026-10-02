"""MTA-SFL Distributed Double DQN Agent — Algorithm 2, Eqs. 37–43.

Implements a per-primary-UAV DDQN agent for relay UAV selection with:
- Normalized state construction from 5 physical features per candidate (Eq. 37)
- Pseudo-count exploration bonus (Eqs. 40–41)
- Masked ε-greedy action selection over feasible relays (Eq. 38)
- Double DQN Bellman update with masked target (Eqs. 42–43)
- Replay buffer with next-step feasibility masks

References
----------
  Eq. 37:  State space construction.
  Eq. 38:  Feasible action space.
  Eq. 39:  Extrinsic reward (see cooperative_reward.py).
  Eq. 40:  Pseudo-count exploration bonus.
  Eq. 41:  Total reward = r_ext + r_int.
  Eq. 42:  Double DQN action selection for target.
  Eq. 43:  Decoupled Bellman target.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ═══════════════════════════════════════════════════════════════════════════
# 1. Q-Network
# ═══════════════════════════════════════════════════════════════════════════

class QNetwork(nn.Module):
    """Multi-layer Q-network mapping state to per-action Q-values.

    Architecture: state_dim → hidden[0] → hidden[1] → … → num_actions

    Args:
        state_dim: Dimensionality of the input state vector.
        num_actions: Number of discrete actions (candidate relays).
        hidden_sizes: Tuple of hidden layer widths.
    """

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        hidden_sizes: Tuple[int, ...] = (256, 128),
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = state_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_actions))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: State tensor of shape ``(batch, state_dim)`` or ``(state_dim,)``.

        Returns:
            Q-values of shape ``(batch, num_actions)`` or ``(num_actions,)``.
        """
        return self.network(x)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Replay Buffer
# ═══════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    """Fixed-capacity experience replay buffer for DDQN training.

    Stores transitions (s, a, r, s', mask', done) as numpy arrays.

    Args:
        capacity: Maximum number of transitions to store.
    """

    def __init__(self, capacity: int = 50000) -> None:
        self._buffer: Deque[Tuple[np.ndarray, int, float,
                                   np.ndarray, np.ndarray, bool]] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_mask: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition.

        Args:
            state: Current state vector.
            action: Selected action index.
            reward: Total reward (r_ext + r_int).
            next_state: Next state vector.
            next_mask: Feasibility mask for next state.
            done: Whether episode terminated.
        """
        self._buffer.append((
            np.asarray(state, dtype=np.float32),
            action,
            reward,
            np.asarray(next_state, dtype=np.float32),
            np.asarray(next_mask, dtype=np.float32),
            done,
        ))

    def sample(
        self, batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        """Uniformly sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Tuple of (states, actions, rewards, next_states, next_masks, dones)
            as torch Tensors.
        """
        batch = random.sample(list(self._buffer), batch_size)
        states, actions, rewards, next_states, next_masks, dones = zip(*batch)

        return (
            torch.from_numpy(np.stack(states)),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.from_numpy(np.stack(next_states)),
            torch.from_numpy(np.stack(next_masks)),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self._buffer)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Pseudo-Count Explorer (Eqs. 40–41)
# ═══════════════════════════════════════════════════════════════════════════

class PseudoCountExplorer:
    """Continuous-to-discrete state hashing for pseudo-count exploration.

    Discretizes each feature into B bins and tracks (state_hash, action)
    visit counts. Returns intrinsic reward r_int = β / √N̂(s, a).

    Args:
        num_features: Dimensionality of the state vector.
        num_bins: Number of quantization bins per feature.
        beta: Exploration coefficient β.
    """

    def __init__(
        self,
        num_features: int,
        num_bins: int = 10,
        beta: float = 0.1,
    ) -> None:
        self.num_features: int = num_features
        self.num_bins: int = num_bins
        self.beta: float = beta
        self._counts: Dict[Tuple, int] = defaultdict(int)

    def discretize(self, state: np.ndarray) -> Tuple:
        """Bin continuous features into discrete indices.

        Each feature assumed ∈ [0, 1]. Values outside are clipped.

        Args:
            state: Continuous state vector.

        Returns:
            Tuple of bin indices.
        """
        clipped = np.clip(state, 0.0, 1.0 - 1e-9)
        bins = (clipped * self.num_bins).astype(int)
        return tuple(bins.tolist())

    def get_bonus(self, state: np.ndarray, action: int) -> float:
        """Compute intrinsic exploration reward.

        r_int = β / √N̂(s, a)

        Args:
            state: Continuous state vector.
            action: Selected action index.

        Returns:
            Intrinsic reward bonus.
        """
        key = (self.discretize(state), action)
        count = self._counts.get(key, 0)
        return self.beta / math.sqrt(count + 1)

    def update(self, state: np.ndarray, action: int) -> None:
        """Increment visit count for (discretized state, action).

        Args:
            state: Continuous state vector.
            action: Selected action index.
        """
        key = (self.discretize(state), action)
        self._counts[key] += 1

    @property
    def total_unique_visits(self) -> int:
        """Number of unique (state, action) pairs visited."""
        return len(self._counts)


# ═══════════════════════════════════════════════════════════════════════════
# 4. DDQN Agent
# ═══════════════════════════════════════════════════════════════════════════

class DDQNAgent:
    """Distributed Double DQN agent for relay selection.

    Each primary UAV runs one DDQNAgent instance. The agent:
    1. Constructs a normalized state from per-relay physical features.
    2. Selects relays via masked ε-greedy.
    3. Computes pseudo-count exploration bonuses.
    4. Trains via Double DQN with Huber loss and gradient clipping.

    Args:
        num_uavs: Number of candidate relay UAVs (= action space size).
        state_features_per_uav: Features per candidate (default 5: CC, E, SINR, L_life, Q).
        lr: Learning rate.
        gamma: Discount factor.
        epsilon_start: Initial exploration rate.
        epsilon_end: Minimum exploration rate.
        epsilon_decay: Multiplicative decay per step.
        target_update_freq: Steps between target network hard updates.
        batch_size: Training batch size.
        buffer_capacity: Replay buffer capacity.
        beta_exploration: Pseudo-count exploration coefficient β.
        num_bins: Bins per feature for pseudo-count hashing.
        device: ``'cpu'`` or ``'cuda'``.
    """

    def __init__(
        self,
        num_uavs: int,
        state_features_per_uav: int = 5,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        target_update_freq: int = 100,
        batch_size: int = 64,
        buffer_capacity: int = 50000,
        beta_exploration: float = 0.1,
        num_bins: int = 10,
        device: str = "cpu",
    ) -> None:
        self.num_uavs: int = num_uavs
        self.state_features_per_uav: int = state_features_per_uav
        self.state_dim: int = state_features_per_uav * num_uavs
        self.gamma: float = gamma
        self.epsilon: float = epsilon_start
        self.epsilon_end: float = epsilon_end
        self.epsilon_decay: float = epsilon_decay
        self.target_update_freq: int = target_update_freq
        self.batch_size: int = batch_size
        self.device: torch.device = torch.device(device)

        # Networks
        self.online_net: QNetwork = QNetwork(
            self.state_dim, num_uavs, hidden_sizes=(256, 128),
        ).to(self.device)
        self.target_net: QNetwork = QNetwork(
            self.state_dim, num_uavs, hidden_sizes=(256, 128),
        ).to(self.device)
        self.update_target_network()

        # Optimizer
        self.optimizer: optim.Adam = optim.Adam(
            self.online_net.parameters(), lr=lr,
        )

        # Replay buffer
        self.replay_buffer: ReplayBuffer = ReplayBuffer(buffer_capacity)

        # Pseudo-count explorer
        self.explorer: PseudoCountExplorer = PseudoCountExplorer(
            self.state_dim, num_bins, beta_exploration,
        )

        # Training counter
        self._step_count: int = 0
        self._train_count: int = 0

    # ----------------------------------------------------------------
    # State Construction (Eq. 37)
    # ----------------------------------------------------------------

    def build_state(
        self,
        cc_values: np.ndarray,
        e_residual_fractions: np.ndarray,
        sinr_values: np.ndarray,
        link_lifetimes: np.ndarray,
        queue_loads: np.ndarray,
        sinr_max_clip: float = 1000.0,
        lifetime_max_clip: float = 100.0,
        queue_max_clip: float = 1e8,
    ) -> np.ndarray:
        """Construct the normalized state vector for all relay candidates.

        s_t = [CC_u, Ẽ_res, SINR̃, L̃_life, Q̃_load]_{u=1}^{|U|}

        All features are normalized to [0, 1].

        Args:
            cc_values: Clustering coefficients per UAV, shape (num_uavs,).
            e_residual_fractions: E_res/E_init per UAV, shape (num_uavs,).
            sinr_values: Raw linear SINR values, shape (num_uavs,).
            link_lifetimes: Predicted route lifetimes (s), shape (num_uavs,).
            queue_loads: Queue loads in bits, shape (num_uavs,).
            sinr_max_clip: SINR normalization ceiling (linear).
            lifetime_max_clip: Lifetime normalization ceiling (s).
            queue_max_clip: Queue load normalization ceiling (bits).

        Returns:
            Flat normalized state vector of shape (5 * num_uavs,).
        """
        cc_norm = np.clip(cc_values, 0.0, 1.0)
        e_norm = np.clip(e_residual_fractions, 0.0, 1.0)
        sinr_norm = np.clip(sinr_values / (sinr_max_clip + 1e-8), 0.0, 1.0)

        # Handle inf in lifetimes (replace with max clip)
        lt = np.where(np.isinf(link_lifetimes), lifetime_max_clip, link_lifetimes)
        lt_norm = np.clip(lt / (lifetime_max_clip + 1e-8), 0.0, 1.0)

        q_norm = np.clip(queue_loads / (queue_max_clip + 1e-8), 0.0, 1.0)

        # Interleave: [CC_0, E_0, SINR_0, L_0, Q_0, CC_1, E_1, ...]
        state = np.zeros(self.state_dim, dtype=np.float32)
        for i in range(self.num_uavs):
            base = i * self.state_features_per_uav
            state[base + 0] = cc_norm[i]
            state[base + 1] = e_norm[i]
            state[base + 2] = sinr_norm[i]
            state[base + 3] = lt_norm[i]
            state[base + 4] = q_norm[i]

        return state

    # ----------------------------------------------------------------
    # Action Selection (Masked ε-greedy)
    # ----------------------------------------------------------------

    def select_action(
        self,
        state: np.ndarray,
        feasibility_mask: np.ndarray,
    ) -> Tuple[int, bool]:
        """Select a relay action using masked ε-greedy.

        Args:
            state: Normalized state vector.
            feasibility_mask: Boolean mask, True = feasible.

        Returns:
            Tuple of (action_index, was_exploration).
            Returns (-1, False) if no feasible action exists.
        """
        feasible_indices = np.where(feasibility_mask)[0]

        if len(feasible_indices) == 0:
            return -1, False

        # ε-greedy with masking
        if random.random() < self.epsilon:
            # Explore: uniform random among feasible
            action = int(np.random.choice(feasible_indices))
            return action, True
        else:
            # Exploit: argmax Q over feasible actions
            state_tensor = torch.from_numpy(
                state.astype(np.float32)
            ).unsqueeze(0).to(self.device)

            with torch.no_grad():
                q_values = self.online_net(state_tensor).squeeze(0)

            # Mask infeasible actions with -inf
            q_masked = q_values.clone()
            mask_tensor = torch.from_numpy(
                feasibility_mask.astype(np.float32)
            ).to(self.device)
            q_masked[mask_tensor == 0] = -float("inf")

            action = int(q_masked.argmax().item())
            return action, False

    # ----------------------------------------------------------------
    # Transition Storage
    # ----------------------------------------------------------------

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_mask: np.ndarray,
        done: bool,
    ) -> None:
        """Push a transition to the replay buffer.

        Also updates the pseudo-count visit tracker.

        Args:
            state: Current state.
            action: Selected action.
            reward: Total reward.
            next_state: Next state.
            next_mask: Next feasibility mask.
            done: Terminal flag.
        """
        self.replay_buffer.push(state, action, reward, next_state, next_mask, done)
        self.explorer.update(state, action)
        self._step_count += 1

        # Periodic target network update
        if self._step_count % self.target_update_freq == 0:
            self.update_target_network()

    # ----------------------------------------------------------------
    # Training Step (Double DQN, Eqs. 42–43)
    # ----------------------------------------------------------------

    def train_step(self) -> Optional[float]:
        """Execute one training step.

        Samples a batch, computes the Double DQN loss with masked
        Bellman targets, and performs a gradient update.

        Returns:
            Loss value, or None if buffer is too small.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, next_masks, dones = \
            self.replay_buffer.sample(self.batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        next_masks = next_masks.to(self.device)
        dones = dones.to(self.device)

        # --- Current Q-values ---
        q_current = self.online_net(states)  # (batch, num_actions)
        q_values = q_current.gather(1, actions.unsqueeze(1)).squeeze(1)

        # --- Double DQN target (Eqs. 42–43) ---
        with torch.no_grad():
            # Online network selects best action under mask
            q_online_next = self.online_net(next_states)  # (batch, num_actions)
            # Mask infeasible actions
            q_online_masked = q_online_next.clone()
            q_online_masked[next_masks == 0] = -float("inf")

            # Check if all actions are masked (routing failure for next state)
            all_masked = (next_masks.sum(dim=1) == 0)

            best_actions = q_online_masked.argmax(dim=1)  # (batch,)

            # Target network evaluates the selected actions
            q_target_next = self.target_net(next_states)
            q_target_values = q_target_next.gather(
                1, best_actions.unsqueeze(1)
            ).squeeze(1)

            # Zero out target for terminal or all-masked states
            q_target_values[dones == 1.0] = 0.0
            q_target_values[all_masked] = 0.0

            targets = rewards + self.gamma * q_target_values

        # --- Huber loss ---
        loss = F.smooth_l1_loss(q_values, targets)

        # --- Backprop with gradient clipping ---
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self._train_count += 1
        return float(loss.item())

    # ----------------------------------------------------------------
    # Target Network Update
    # ----------------------------------------------------------------

    def update_target_network(self) -> None:
        """Hard copy online network weights to target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ----------------------------------------------------------------
    # Epsilon Decay
    # ----------------------------------------------------------------

    def decay_epsilon(self) -> float:
        """Decay epsilon by multiplicative factor.

        Returns:
            New epsilon value.
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return self.epsilon

    # ----------------------------------------------------------------
    # Exploration Bonus
    # ----------------------------------------------------------------

    def get_exploration_bonus(self, state: np.ndarray, action: int) -> float:
        """Get the pseudo-count exploration bonus for a (state, action) pair.

        Args:
            state: Current state vector.
            action: Selected action index.

        Returns:
            Intrinsic reward r_int = β / √N̂(s, a).
        """
        return self.explorer.get_bonus(state, action)

    # ----------------------------------------------------------------
    # Statistics
    # ----------------------------------------------------------------

    def get_exploration_stats(self) -> Dict[str, float]:
        """Return current exploration statistics.

        Returns:
            Dict with epsilon, buffer_size, train_count, step_count,
            unique_state_action_pairs.
        """
        return {
            "epsilon": self.epsilon,
            "buffer_size": float(len(self.replay_buffer)),
            "train_count": float(self._train_count),
            "step_count": float(self._step_count),
            "unique_state_action_pairs": float(self.explorer.total_unique_visits),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Running ddqn_agent.py self-tests …")

    NUM_UAVS = 6
    agent = DDQNAgent(
        num_uavs=NUM_UAVS,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.99,
        target_update_freq=50,
        batch_size=16,
        buffer_capacity=1000,
        beta_exploration=0.1,
        num_bins=10,
        device="cpu",
    )

    # ---- Test 1: State construction ----
    cc = np.array([0.5, 0.3, 0.8, 0.1, 0.9, 0.6])
    e_frac = np.array([0.9, 0.7, 0.5, 0.3, 0.1, 0.95])
    sinr = np.array([100.0, 500.0, 50.0, 10.0, 800.0, 200.0])
    lifetimes = np.array([50.0, 30.0, float("inf"), 5.0, 80.0, 15.0])
    queues = np.array([1e6, 5e7, 0.0, 1e8, 2e7, 1e4])

    state = agent.build_state(cc, e_frac, sinr, lifetimes, queues)
    assert state.shape == (30,), f"State shape = {state.shape}, expected (30,)"
    assert np.all(state >= 0.0) and np.all(state <= 1.0), \
        f"State values outside [0,1]: min={state.min()}, max={state.max()}"
    print(f"  [OK] State shape = {state.shape}, all values ∈ [0, 1]")

    # ---- Test 2: Masked action selection ----
    mask = np.array([False, True, True, False, False, True])
    feasible_set = set(np.where(mask)[0])

    for _ in range(50):
        action, explored = agent.select_action(state, mask)
        assert action in feasible_set, \
            f"Action {action} not in feasible set {feasible_set}"
    print("  [OK] All 50 actions selected from feasible set")

    # All-masked case
    empty_mask = np.zeros(NUM_UAVS, dtype=bool)
    action_empty, _ = agent.select_action(state, empty_mask)
    assert action_empty == -1, f"Expected -1 for empty mask, got {action_empty}"
    print("  [OK] Empty mask returns action = -1")

    # ---- Test 3: Pseudo-count bonus ----
    bonus_1 = agent.get_exploration_bonus(state, 1)
    agent.explorer.update(state, 1)
    bonus_2 = agent.get_exploration_bonus(state, 1)
    assert bonus_2 < bonus_1, \
        f"Bonus should decrease: {bonus_1:.4f} → {bonus_2:.4f}"
    print(f"  [OK] Pseudo-count bonus decreases: {bonus_1:.4f} → {bonus_2:.4f}")

    # ---- Test 4: Store transitions and train ----
    # Fill buffer with synthetic transitions
    for i in range(100):
        s = agent.build_state(
            np.random.rand(NUM_UAVS),
            np.random.rand(NUM_UAVS),
            np.random.rand(NUM_UAVS) * 500,
            np.random.rand(NUM_UAVS) * 100,
            np.random.rand(NUM_UAVS) * 1e8,
        )
        m = np.random.rand(NUM_UAVS) > 0.3
        if not np.any(m):
            m[0] = True
        a, _ = agent.select_action(s, m)
        r = np.random.randn()
        s2 = agent.build_state(
            np.random.rand(NUM_UAVS),
            np.random.rand(NUM_UAVS),
            np.random.rand(NUM_UAVS) * 500,
            np.random.rand(NUM_UAVS) * 100,
            np.random.rand(NUM_UAVS) * 1e8,
        )
        m2 = np.random.rand(NUM_UAVS) > 0.3
        if not np.any(m2):
            m2[0] = True
        done = i == 99
        agent.store_transition(s, a, r, s2, m2, done)

    assert len(agent.replay_buffer) == 100
    print(f"  [OK] Buffer filled: {len(agent.replay_buffer)} transitions")

    # ---- Test 5: Training loop ----
    losses: List[float] = []
    for step in range(200):
        loss = agent.train_step()
        if loss is not None:
            assert math.isfinite(loss), f"Non-finite loss at step {step}: {loss}"
            losses.append(loss)
        agent.decay_epsilon()

    assert len(losses) > 0, "Should have trained at least once"
    print(f"  [OK] {len(losses)} training steps, all losses finite")
    print(f"       Loss range: [{min(losses):.6f}, {max(losses):.6f}]")

    # ---- Test 6: Target network update ----
    agent.update_target_network()
    for p_online, p_target in zip(
        agent.online_net.parameters(), agent.target_net.parameters()
    ):
        assert torch.allclose(p_online, p_target), "Target should match online"
    print("  [OK] Target network matches online network after update")

    # ---- Test 7: Epsilon decay ----
    stats = agent.get_exploration_stats()
    assert stats["epsilon"] < 1.0, "Epsilon should have decayed"
    assert stats["epsilon"] >= 0.05, "Epsilon should not go below minimum"
    print(f"  [OK] Epsilon = {stats['epsilon']:.4f} (decayed)")

    # ---- Test 8: Exploration stats ----
    assert stats["buffer_size"] == 100
    assert stats["train_count"] > 0
    assert stats["unique_state_action_pairs"] > 0
    print(f"  [OK] Stats: {stats}")

    print("\n[PASS] ddqn_agent.py — All assertions passed.")
