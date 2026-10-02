"""MTA-SFL Topology Stability Tracker.

Tracks cluster-head churn, topology repair overhead, and depletion adaptation
to address Reviewer 1's topology stability critique.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class RepairEvent:
    """Simulated time/energy penalty for emergency topology repair."""
    round_idx: int
    primary_uav: int
    time_penalty_s: float
    energy_penalty_j: float


class TopologyStabilityTracker:
    """Calculates cluster-head churn, repair overhead, and topological adaptation."""
    
    def __init__(self) -> None:
        self.previous_relays: Dict[int, int] = {}
        self.total_rounds: int = 0
        self.churn_events: int = 0
        self.switch_opportunities: int = 0
        self.repairs: List[RepairEvent] = []
        self.bridge_activations: int = 0
        
    def log_round(self, primary_uav: int, selected_relay: int, is_bridge: bool) -> None:
        """Log cluster-head selection for the round to detect churn."""
        if primary_uav in self.previous_relays:
            # If it's already in the dictionary, this is a subsequent round, meaning it has an opportunity to switch.
            if selected_relay != -1 and self.previous_relays[primary_uav] != -1:
                self.switch_opportunities += 1
                if self.previous_relays[primary_uav] != selected_relay:
                    self.churn_events += 1
                
        if selected_relay != -1:
            self.previous_relays[primary_uav] = selected_relay
            
        if is_bridge:
            self.bridge_activations += 1
            
    def log_repair(self, round_idx: int, primary_uav: int, time_pen_s: float, energy_pen_j: float) -> None:
        """Log an emergency discovery broadcast due to route failure."""
        self.repairs.append(RepairEvent(round_idx, primary_uav, time_pen_s, energy_pen_j))
        
    def step_round(self) -> None:
        """Advance the round counter."""
        self.total_rounds += 1
        
    def get_churn_rate(self) -> float:
        """CR = Total Relay Switches / Total Switch Opportunities"""
        if self.switch_opportunities == 0:
            return 0.0
        return self.churn_events / self.switch_opportunities
        
    def get_avg_repair_penalties(self) -> Dict[str, float]:
        """Average repair time and energy penalty per repair event."""
        if not self.repairs:
            return {"time_s": 0.0, "energy_j": 0.0, "count": 0.0}
            
        avg_t = sum(r.time_penalty_s for r in self.repairs) / len(self.repairs)
        avg_e = sum(r.energy_penalty_j for r in self.repairs) / len(self.repairs)
        
        return {"time_s": avg_t, "energy_j": avg_e, "count": float(len(self.repairs))}
        
if __name__ == "__main__":
    print("Running topology_tracker.py self-tests ...")
    
    tracker = TopologyStabilityTracker()
    
    # Simulate stable rounds
    for _ in range(5):
        tracker.step_round()
        tracker.log_round(primary_uav=1, selected_relay=3, is_bridge=False)
        
    assert tracker.get_churn_rate() == 0.0
    print("  [OK] Stable topology gives 0.0 churn rate")
    
    # Simulate a switch (churn event)
    tracker.step_round()
    tracker.log_round(primary_uav=1, selected_relay=4, is_bridge=False)
    
    # Total opportunities = 5 (6 rounds total, first round doesn't count), switches = 1 -> Churn Rate = 1/5
    cr = tracker.get_churn_rate()
    assert abs(cr - (1.0 / 5.0)) < 1e-6
    print(f"  [OK] Correct churn rate upon switch: {cr:.4f}")
    
    # Simulate a repair
    tracker.log_repair(6, 1, time_pen_s=0.05, energy_pen_j=2.0)
    tracker.log_repair(7, 1, time_pen_s=0.15, energy_pen_j=4.0)
    pen = tracker.get_avg_repair_penalties()
    assert abs(pen["time_s"] - 0.10) < 1e-6
    assert abs(pen["energy_j"] - 3.0) < 1e-6
    print("  [OK] Correct repair penalty averages")
    
    # Simulate bridge
    tracker.log_round(primary_uav=2, selected_relay=5, is_bridge=True)
    assert tracker.bridge_activations == 1
    print("  [OK] Bridge activation logged")
    
    print("[PASS] topology_tracker.py — All assertions passed.")
