"""流量标定与标定有效期。

流量计的读数会随结垢与检修漂移，标定给出新的基线，联锁门限必须立刻按新基线
重算。标定记录保留代际、标定值、当时生效的容差与操作人，便于巡检比对"哪次
标定之后联锁开始误动"。

两条运行期规则：

* **容差自适应**：稳态运行的实测流量的波动（σ）持续估计，容差按
  ``σ × 倍数`` 在配置上下限内自动收紧或放宽，不再用拍脑袋的定值。
* **有效期按实测偏差算（定死，不按挂钟时间）**：标定后只要实测流量与基线
  的偏差持续越带超过确认窗口，这次标定即作废。工况漂移是偏差现象，按时间
  到期会误杀仍然有效的标定、放过已经漂移的标定。作废后的处置（退回上一次
  有效标定，或停用联锁等人工标定）按回路配置
  ``SaltLimits.interlock_on_invalidation`` 决定；出厂基线（代际 0）本身
  被作废时已无上一次可退，一律转入停用等人工标定。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import INVALIDATION_HOLDOUT, SaltLimits
from ..receiver.threshold import ThresholdModel

# 作废 / 回退 / 停用事件的动作名，平台控制循环按原样转发到审计。
ACTION_VOIDED = "flow_calibration_voided"
ACTION_ROLLBACK = "flow_calibration_rollback"
ACTION_HOLDOUT = "flow_interlock_holdout"


@dataclass
class CalibrationRecord:
    """一次流量标定；作废后保留在台账里并标记原因。"""

    epoch: int
    measured_kg_s: float
    baseline_kg_s: float
    threshold_kg_s: float
    tolerance_kg_s: float
    operator: str
    ts: float
    voided: bool = False
    void_reason: str = ""

    def mark_void(self, reason: str) -> None:
        """把这次标定标记为作废。"""

        self.voided = True
        self.void_reason = str(reason)

    def as_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "measured_kg_s": round(self.measured_kg_s, 3),
            "baseline_kg_s": round(self.baseline_kg_s, 3),
            "threshold_kg_s": round(self.threshold_kg_s, 3),
            "tolerance_kg_s": round(self.tolerance_kg_s, 3),
            "operator": self.operator,
            "ts": round(self.ts, 3),
            "voided": self.voided,
            "void_reason": self.void_reason,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "CalibrationRecord":
        """从落盘字段还原一条标定记录。"""

        return cls(
            epoch=int(raw["epoch"]),
            measured_kg_s=float(raw["measured_kg_s"]),
            baseline_kg_s=float(raw["baseline_kg_s"]),
            threshold_kg_s=float(raw["threshold_kg_s"]),
            tolerance_kg_s=float(raw["tolerance_kg_s"]),
            operator=str(raw.get("operator", "")),
            ts=float(raw.get("ts", 0.0)),
            voided=bool(raw.get("voided", False)),
            void_reason=str(raw.get("void_reason", "")),
        )


class FlowCalibrator:
    """把现场实测流量写回联锁基线，并看守标定有效期。"""

    def __init__(self, limits: SaltLimits, threshold: ThresholdModel) -> None:
        self._limits = limits
        self._threshold = threshold
        self._records: list[CalibrationRecord] = []
        # 运行波动的在线估计（Welford）：样本数、均值、离差平方和。
        self._samples = 0
        self._mean = 0.0
        self._m2 = 0.0
        # 实测偏差越带的累计时长，回到带内即清零。
        self._violation_s = 0.0

    @property
    def limits(self) -> SaltLimits:
        return self._limits

    @property
    def threshold(self) -> ThresholdModel:
        return self._threshold

    def calibrate(self, measured_kg_s: float, now: float, operator: str = "console") -> CalibrationRecord:
        """用实测流量重新标定基线，并同步刷新联锁门限。"""

        measured = float(measured_kg_s)
        if measured <= self._threshold.tolerance_kg_s:
            raise ValueError("实测流量必须大于当前流量容差")
        baseline = self._threshold.calibrate(measured)
        self._violation_s = 0.0
        record = CalibrationRecord(
            epoch=self._threshold.epoch,
            measured_kg_s=measured,
            baseline_kg_s=baseline,
            threshold_kg_s=self._threshold.threshold_kg_s(),
            tolerance_kg_s=self._threshold.tolerance_kg_s,
            operator=str(operator),
            ts=float(now),
        )
        self._records.append(record)
        return record

    def observe(self, flow_kg_s: float, dt: float) -> list[dict]:
        """吃进一个稳态流量样本：推进容差自适应，并看守标定有效期。

        只在稳态循环（单泵在运且已到目标流量）时被调用。返回本样本触发的
        事件列表（作废／回退／停用），无事发生则为空表。
        """

        if dt <= 0:
            raise ValueError("步长必须为正")
        self._track_fluctuation(float(flow_kg_s))
        if not self._threshold.active:
            return []
        deviation = abs(float(flow_kg_s) - self._threshold.baseline_kg_s)
        if deviation <= self._limits.calibration_max_deviation_kg_s:
            self._violation_s = 0.0
            return []
        self._violation_s += float(dt)
        if self._violation_s < self._limits.calibration_confirm_s:
            return []
        self._violation_s = 0.0
        return self._invalidate(deviation)

    def _track_fluctuation(self, flow_kg_s: float) -> None:
        """更新波动估计，并按当前 σ 调整容差（样本不足时保持出厂容差）。"""

        self._samples += 1
        delta = flow_kg_s - self._mean
        self._mean += delta / self._samples
        self._m2 += delta * (flow_kg_s - self._mean)
        if self._samples >= self._limits.calibration_min_samples:
            self._threshold.set_tolerance(self.sigma_kg_s() * self._limits.flow_tolerance_sigma_mult)

    def _invalidate(self, deviation_kg_s: float) -> list[dict]:
        """作废当前标定，并按回路策略回退或停用联锁。"""

        voided_epoch = self._threshold.epoch
        voided_baseline = self._threshold.baseline_kg_s
        reason = (
            f"实测偏差 {deviation_kg_s:.3f} kg/s 持续越带 "
            f"(±{self._limits.calibration_max_deviation_kg_s} kg/s)"
        )
        record = self._record_for(voided_epoch)
        if record is not None:
            record.mark_void(reason)
        events = [
            {
                "action": ACTION_VOIDED,
                "message": f"标定代际 {voided_epoch} 作废：{reason}",
                "epoch": voided_epoch,
                "baseline_kg_s": round(voided_baseline, 3),
                "deviation_kg_s": round(deviation_kg_s, 3),
            }
        ]
        if self._limits.interlock_on_invalidation == INVALIDATION_HOLDOUT or voided_epoch == 0:
            # 策略要求停用，或出厂基线（代际 0）本身被作废、已无上一次可退。
            self._threshold.holdout(reason)
            events.append(
                {
                    "action": ACTION_HOLDOUT,
                    "message": "低流量联锁已停用，等待人工标定",
                    "reason": reason,
                    "epoch": voided_epoch,
                }
            )
            return events
        target = self._rollback_target()
        restored = self._threshold.restore(target[0], target[1])
        events.append(
            {
                "action": ACTION_ROLLBACK,
                "message": f"联锁基线回退到标定代际 {target[1]}",
                "from_epoch": voided_epoch,
                "epoch": target[1],
                "baseline_kg_s": round(restored, 3),
            }
        )
        return events

    def _rollback_target(self) -> tuple[float, int]:
        """找退回目标：最近一次仍有效的标定；没有则退回出厂基线（代际 0）。"""

        for record in reversed(self._records):
            if not record.voided:
                return record.baseline_kg_s, record.epoch
        return self._limits.flow_baseline_kg_s, 0

    def _record_for(self, epoch: int) -> CalibrationRecord | None:
        """按代际找标定记录。"""

        for record in reversed(self._records):
            if record.epoch == epoch:
                return record
        return None

    def records(self) -> list[CalibrationRecord]:
        """返回全部标定记录。"""

        return list(self._records)

    def restore_records(self, records: list[CalibrationRecord]) -> None:
        """把落盘的标定台账整体装回（重启恢复用），并接续代际序号。"""

        self._records = list(records)
        for record in self._records:
            self._threshold.note_epoch(record.epoch)

    def last(self) -> CalibrationRecord | None:
        """返回最近一次标定记录。"""

        return self._records[-1] if self._records else None

    def baseline_kg_s(self) -> float:
        """返回当前基线。"""

        return self._threshold.baseline_kg_s

    def threshold_kg_s(self) -> float:
        """返回当前门限。"""

        return self._threshold.threshold_kg_s()

    def tolerance_kg_s(self) -> float:
        """返回当前生效的自适应容差。"""

        return self._threshold.tolerance_kg_s

    def sigma_kg_s(self) -> float:
        """返回运行波动的当前估计（样本标准差）。"""

        if self._samples < 2:
            return 0.0
        return (self._m2 / (self._samples - 1)) ** 0.5

    def fluctuation(self) -> dict:
        """返回波动估计的可打印形式。"""

        return {
            "samples": self._samples,
            "mean_kg_s": round(self._mean, 3),
            "sigma_kg_s": round(self.sigma_kg_s(), 3),
        }
