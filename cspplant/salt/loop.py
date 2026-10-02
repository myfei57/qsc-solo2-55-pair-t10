"""熔盐回路控制器。

把罐、泵、水力模型、标定与持久化串成一个回路：循环时推进两台泵并汇总净流量，
同时按流量增减冷热罐存量；切换走 :class:`~cspplant.salt.handoff.TankHandoff`
的固定顺序；标定走 :class:`~cspplant.salt.calibrate.FlowCalibrator`，标定结果
直接改变联锁门限。
"""

from __future__ import annotations

import threading

from ..config import SaltLimits
from ..errors import InterlockError
from ..receiver.header import SaltHeader
from ..receiver.thermal import ThermalModel
from ..receiver.threshold import Generation, Migration, ThresholdModel
from .calibrate import FlowCalibrator
from .handoff import HandoffReport, TankHandoff
from .hydraulic import HydraulicLoop
from .persist import SaltPersistence
from .pump import SaltPump
from .snapshot import SaltSnapshot
from .tank import SaltTank

COLD_TANK_CAPACITY_KG = 900_000.0
HOT_TANK_CAPACITY_KG = 900_000.0


def _round_or_none(value: float | None) -> float | None:
    return None if value is None else round(value, 3)


class SaltLoop:
    """冷热罐、盐泵与母管的回路控制器。"""

    def __init__(
        self,
        limits: SaltLimits,
        threshold: ThresholdModel,
        thermal: ThermalModel,
        header: SaltHeader,
        persistence: SaltPersistence,
        now: float = 0.0,
    ) -> None:
        self._limits = limits
        self._threshold = threshold
        self._thermal = thermal
        self._header = header
        self._persistence = persistence
        self._cold_tank = SaltTank("cold-tank", COLD_TANK_CAPACITY_KG, COLD_TANK_CAPACITY_KG * 0.82)
        self._hot_tank = SaltTank("hot-tank", HOT_TANK_CAPACITY_KG, HOT_TANK_CAPACITY_KG * 0.12)
        self._cold_pump = SaltPump("cold-pump", limits)
        self._hot_pump = SaltPump("hot-pump", limits)
        self._hydraulic = HydraulicLoop(limits)
        self._handoff = TankHandoff(limits)
        self._calibrator = FlowCalibrator(limits, threshold)
        self._lock = threading.RLock()
        self._last_now = float(now)
        self._restore()

    def _restore(self) -> None:
        """重启后恢复罐位、流量基线、标定代际与换代迁移。"""

        levels = self._persistence.read_levels()
        if levels is not None:
            self._cold_tank.set_level_pct(levels["cold_pct"])
            self._hot_tank.set_level_pct(levels["hot_pct"])

        active = self._persistence.read_active_generation()
        saved_generations = self._persistence.read_calibrations()
        if active is not None and active["baseline_kg_s"] > 0:
            if active.get("calibration_id"):
                self._threshold.restore(
                    calibration_id=active["calibration_id"],
                    epoch=active["epoch"],
                    baseline_kg_s=active["baseline_kg_s"],
                    tolerance_kg_s=active["tolerance_kg_s"]
                    or self._limits.flow_tolerance_kg_s,
                    state=active.get("state") or "confirmed",
                    created_ts=active["created_ts"],
                    confirmed_ts=active.get("confirmed_ts"),
                    expires_ts=active.get("expires_ts"),
                    operator=active.get("operator", "restore"),
                    armed=active.get("armed", True),
                )
            else:
                # 旧版落盘只有基线与代际，按直接落地一代恢复。
                self._threshold.calibrate(active["baseline_kg_s"], now=active["created_ts"])

        # 回放全部代际历史（restore() 已放当前代，这里补齐其余且不重复）。
        active_id = active.get("calibration_id") if active else None
        for raw in saved_generations:
            if str(raw.get("calibration_id")) == str(active_id):
                continue  # 当前代已由 restore() 落地
            self._append_saved_generation(raw)

        # 回放换代迁移：restore() 已自带一条 restore 迁移，落盘的迁移原样补齐，去重。
        existing = {(m.kind, m.ts, m.from_id, m.to_id) for m in self._threshold.migrations()}
        for raw in self._persistence.read_migrations():
            try:
                migration = Migration(
                    from_epoch=raw.get("from_epoch"),
                    to_epoch=raw.get("to_epoch"),
                    from_id=str(raw.get("from_id", "")),
                    to_id=str(raw.get("to_id", "")),
                    kind=str(raw.get("kind", "")),
                    ts=float(raw.get("ts", 0.0)),
                    operator=str(raw.get("operator", "")),
                    reason=str(raw.get("reason", "")),
                )
            except (TypeError, ValueError):
                continue
            key = (migration.kind, migration.ts, migration.from_id, migration.to_id)
            if key not in existing:
                self._threshold.migrations().append(migration)
                existing.add(key)

    def _append_saved_generation(self, raw: dict) -> None:
        """把一条落盘代际记录追加回历史（不改当前指针、不触发换代）。"""

        try:
            generation = Generation(
                calibration_id=str(raw["calibration_id"]),
                epoch=int(raw["epoch"]),
                baseline_kg_s=float(raw["baseline_kg_s"]),
                tolerance_kg_s=float(raw["tolerance_kg_s"]),
                measured_kg_s=float(raw.get("measured_kg_s", raw["baseline_kg_s"])),
                operator=str(raw.get("operator", "restore")),
                created_ts=float(raw.get("created_ts", 0.0)),
                state=str(raw.get("state", "confirmed")),
                confirmed_ts=raw.get("confirmed_ts"),
                expires_ts=raw.get("expires_ts"),
                source=str(raw.get("source", "restore")),
                reject_reason=str(raw.get("reject_reason", "")),
            )
        except (KeyError, TypeError, ValueError):
            return
        self._threshold.append_history(generation)

    @property
    def limits(self) -> SaltLimits:
        return self._limits

    @property
    def cold_tank(self) -> SaltTank:
        return self._cold_tank

    @property
    def hot_tank(self) -> SaltTank:
        return self._hot_tank

    @property
    def cold_pump(self) -> SaltPump:
        return self._cold_pump

    @property
    def hot_pump(self) -> SaltPump:
        return self._hot_pump

    @property
    def hydraulic(self) -> HydraulicLoop:
        return self._hydraulic

    @property
    def handoff(self) -> TankHandoff:
        return self._handoff

    @property
    def calibrator(self) -> FlowCalibrator:
        return self._calibrator

    @property
    def threshold(self) -> ThresholdModel:
        return self._threshold

    def start_cold(self) -> float:
        """启冷泵。"""

        with self._lock:
            if self._cold_tank.at_min():
                raise InterlockError("冷盐罐已到下限，禁止启冷泵", component="salt")
            return self._cold_pump.start()

    def stop_cold(self) -> float:
        """停冷泵。"""

        with self._lock:
            return self._cold_pump.stop()

    def start_hot(self) -> float:
        """启热泵；冷泵仍在运转即拒绝。"""

        with self._lock:
            if self._cold_pump.running():
                raise InterlockError("冷盐泵仍在运行，禁止启热盐泵", component="salt")
            return self._hot_pump.start()

    def stop_hot(self) -> float:
        """停热泵。"""

        with self._lock:
            return self._hot_pump.stop()

    def pumps_running(self) -> int:
        """返回在运泵数量。"""

        return sum(1 for pump in (self._cold_pump, self._hot_pump) if pump.running())

    def flow_kg_s(self) -> float:
        """返回最近一次汇总的净流量。"""

        return self._hydraulic.net_flow_kg_s()

    def step(self, dt: float, now: float | None = None) -> dict:
        """推进一个循环步：泵升速、净流量汇总、罐位记账、母管压力更新、标定观测。"""

        if dt <= 0:
            raise ValueError("步长必须为正")
        with self._lock:
            flow = self._hydraulic.update([self._cold_pump, self._hot_pump], dt)
            moved = 0.0
            if flow > 0.0:
                try:
                    moved = self._cold_tank.draw(flow * dt)
                    self._hot_tank.fill(moved)
                except InterlockError:
                    moved = 0.0
            pressure = self._header.update(flow, self.pumps_running(), dt)
            # 在运实测喂给标定器：累计波动 + 推进候选标定确认窗。
            moment = float(self._last_now if now is None else now)
            self._last_now = moment
            change = self._calibrator.observe(flow, self.pumps_running() > 0, moment)
            if change is not None:
                self._persist_calibration()
            self._persistence.save_levels(self._cold_tank.level_pct(), self._hot_tank.level_pct(), moment)
            return {
                "flow_kg_s": round(flow, 3),
                "moved_kg": round(moved, 3),
                "pressure_mpa": round(pressure, 3),
                "pumps_running": self.pumps_running(),
                "overpressure": self._header.overpressure(),
                "calibration_change": change,
            }

    def changeover(self, dt: float = 1.0) -> HandoffReport:
        """执行冷热罐切换。"""

        with self._lock:
            return self._handoff.changeover(
                self._cold_pump,
                self._hot_pump,
                self._cold_tank,
                self._hot_tank,
                self._header,
                self._hydraulic,
                dt,
            )

    def calibrate_flow(self, measured_kg_s: float, now: float, operator: str = "console") -> Generation:
        """开始一次流量标定（临时代），落盘候选基线；正式生效要过确认窗。"""

        with self._lock:
            self._last_now = float(now)
            record = self._calibrator.begin(measured_kg_s, now, operator)
            self._persist_calibration()
            return record

    def confirm_calibration(self, now: float) -> tuple[Generation, Migration | None]:
        """提前收尾确认窗（正常由控制周期自动确认）。"""

        with self._lock:
            self._last_now = float(now)
            generation, migration = self._calibrator.confirm(now)
            self._persist_calibration()
            return generation, migration

    def _persist_calibration(self) -> None:
        """把当前代际、全部标定记录与迁移落盘。"""

        active = self._threshold.current()
        # 当前代际文档（含容差/有效期/投用状态）即基线文档，另存旧口径会覆盖它。
        self._persistence.save_active_generation(active.as_dict(), self._threshold.armed())
        self._persistence.save_calibrations(
            [generation.as_dict() for generation in self._threshold.generations()]
        )
        self._persistence.save_migrations(
            [migration.as_dict() for migration in self._threshold.migrations()]
        )

    def threshold_kg_s(self) -> float:
        """返回当前低流量联锁门限。"""

        return self._threshold.threshold_kg_s()

    def disturb(self, heat_fraction: float) -> dict:
        """注入一次热负荷扰动：热罐多进盐、冷罐多出盐。"""

        with self._lock:
            delta = max(0.0, min(1.0, float(heat_fraction))) * 2_000.0
            self._cold_tank.draw(delta)
            added = self._hot_tank.fill(delta)
            return {
                "moved_kg": round(added, 1),
                "cold_level_pct": round(self._cold_tank.level_pct(), 3),
                "hot_level_pct": round(self._hot_tank.level_pct(), 3),
                "temperature_c": round(self._thermal.value_c(), 2),
            }

    def status(self, now: float | None = None) -> dict:
        """汇总熔盐回路状态。"""

        current = self._threshold.current()
        moment = self._last_now if now is None else float(now)
        return {
            "cold_tank": self._cold_tank.as_dict(),
            "hot_tank": self._hot_tank.as_dict(),
            "cold_pump": self._cold_pump.as_dict(),
            "hot_pump": self._hot_pump.as_dict(),
            "net_flow_kg_s": round(self._hydraulic.net_flow_kg_s(), 3),
            "opposed_steps": self._hydraulic.opposed_steps(),
            "hydraulic_updates": self._hydraulic.updates(),
            "header": self._header.status().as_dict(),
            "header_admits": self._header.admits(),
            "header_overpressure_events": self._header.overpressure_events(),
            "baseline_kg_s": round(self._threshold.baseline_kg_s, 3),
            "threshold_kg_s": round(self._threshold.threshold_kg_s(), 3),
            "tolerance_kg_s": round(self._threshold.tolerance_kg_s, 3),
            "adaptive_tolerance_kg_s": round(self._calibrator.stats.tolerance_kg_s(), 3),
            "flow_sigma_kg_s": _round_or_none(self._calibrator.stats.sigma_kg_s()),
            "flow_samples": self._calibrator.stats.samples(),
            "threshold_evaluations": self._threshold.evaluations(),
            "epoch": self._threshold.epoch,
            "calibration_id": self._threshold.calibration_id,
            "calibration_state": self._threshold.calibration_state(moment),
            "interlock_armed": self._threshold.armed(),
            "calibration_pending": self._threshold.has_pending(),
            "disable_reason": self._threshold.disable_reason(),
            "confirm_window_s": self._limits.calibration_confirm_s,
            "valid_for_s": self._limits.calibration_valid_for_s,
            "invalid_policy": self._limits.calibration_invalid_policy,
            "current_expires_ts": current.expires_ts,
            "changeovers": self._handoff.changeovers(),
            "calibrations": [generation.as_dict() for generation in self.calibrator.records()],
            "migrations": [migration.as_dict() for migration in self.calibrator.migrations()],
        }

    def snapshot(self) -> SaltSnapshot:
        """返回熔盐回路状态快照对象。"""

        return SaltSnapshot(
            cold_level_pct=self._cold_tank.level_pct(),
            hot_level_pct=self._hot_tank.level_pct(),
            cold_flow_kg_s=self._cold_pump.flow_kg_s(),
            hot_flow_kg_s=self._hot_pump.flow_kg_s(),
            net_flow_kg_s=self._hydraulic.net_flow_kg_s(),
            pumps_running=self.pumps_running(),
            opposed_steps=self._hydraulic.opposed_steps(),
            baseline_kg_s=self._threshold.baseline_kg_s,
            threshold_kg_s=self._threshold.threshold_kg_s(),
            epoch=self._threshold.epoch,
            calibration_id=self._threshold.calibration_id,
            calibration_state=self._threshold.calibration_state(self._last_now),
            tolerance_kg_s=self._threshold.tolerance_kg_s,
            interlock_armed=self._threshold.armed(),
        )
