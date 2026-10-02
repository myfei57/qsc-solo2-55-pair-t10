"""熔盐回路状态快照。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SaltSnapshot:
    """熔盐回路某一时刻的状态。"""

    cold_level_pct: float
    hot_level_pct: float
    cold_flow_kg_s: float
    hot_flow_kg_s: float
    net_flow_kg_s: float
    pumps_running: int
    opposed_steps: int
    baseline_kg_s: float
    threshold_kg_s: float
    epoch: int
    calibration_id: str = "cal-seed"
    calibration_state: str = "seed"
    tolerance_kg_s: float = 0.0
    interlock_armed: bool = True

    def as_dict(self) -> dict:
        return {
            "cold_level_pct": round(self.cold_level_pct, 3),
            "hot_level_pct": round(self.hot_level_pct, 3),
            "cold_flow_kg_s": round(self.cold_flow_kg_s, 3),
            "hot_flow_kg_s": round(self.hot_flow_kg_s, 3),
            "net_flow_kg_s": round(self.net_flow_kg_s, 3),
            "pumps_running": self.pumps_running,
            "opposed_steps": self.opposed_steps,
            "baseline_kg_s": round(self.baseline_kg_s, 3),
            "threshold_kg_s": round(self.threshold_kg_s, 3),
            "tolerance_kg_s": round(self.tolerance_kg_s, 3),
            "epoch": self.epoch,
            "calibration_id": self.calibration_id,
            "calibration_state": self.calibration_state,
            "interlock_armed": self.interlock_armed,
        }
