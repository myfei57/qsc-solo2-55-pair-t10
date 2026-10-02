"""低流量联锁门限。

门限不是一个固定值，而是"当前标定基线 - 容差"。标定动过之后门限必须跟着变，
否则联锁会按旧基线判定，正常波动也会被当成低流量。容差也不是定值：运行中按
实测波动自动收紧或放宽（由 :class:`~cspplant.salt.calibrate.FlowCalibrator`
推进，这里只负责夹到配置上下限之内）。基线、容差与代际放在一起，门限只从
基线推导，不允许调用方绕过基线自己给门限。

代际（epoch）标识"哪一次标定"：每次标定取一个新的、不复用的序号；回退到
上一次标定只是把生效代际指回去，不消耗新序号。标定作废后也可以按回路策略
停用联锁（holdout），等人工标定重新投入。
"""

from __future__ import annotations

from ..config import SaltLimits


class ThresholdModel:
    """按标定基线推导低流量联锁门限。"""

    def __init__(self, limits: SaltLimits, baseline_kg_s: float | None = None) -> None:
        self._limits = limits
        self._baseline = float(limits.flow_baseline_kg_s if baseline_kg_s is None else baseline_kg_s)
        self._tolerance = float(limits.flow_tolerance_kg_s)
        self._epoch = 0
        self._next_epoch = 0
        self._active = True
        self._hold_reason = ""
        self._trips = 0
        self._evaluations = 0

    @property
    def limits(self) -> SaltLimits:
        return self._limits

    @property
    def baseline_kg_s(self) -> float:
        return self._baseline

    @property
    def tolerance_kg_s(self) -> float:
        """返回当前生效的流量容差（自适应，已夹到上下限内）。"""

        return self._tolerance

    @property
    def epoch(self) -> int:
        """返回当前生效的标定代际。"""

        return self._epoch

    @property
    def active(self) -> bool:
        """联锁是否投入；停用时判定一律放行并标明 ``active=False``。"""

        return self._active

    @property
    def hold_reason(self) -> str:
        """联锁被停用的原因；未停用时为空串。"""

        return self._hold_reason

    def threshold_kg_s(self) -> float:
        """返回当前联锁门限。"""

        return self._limits.low_flow_threshold(self._baseline, self._tolerance)

    def set_tolerance(self, tolerance_kg_s: float) -> float:
        """按实测波动调整容差，夹到配置上下限内，返回实际生效值。

        上限同时受基线约束：容差再宽也必须给门限留出至少一个下限容差的
        空间，保证门限始终为正。
        """

        low = self._limits.flow_tolerance_min_kg_s
        high = min(self._limits.flow_tolerance_max_kg_s, self._baseline - low)
        high = max(high, low)
        self._tolerance = min(high, max(low, float(tolerance_kg_s)))
        return self._tolerance

    def calibrate(self, baseline_kg_s: float) -> float:
        """重新标定基线并推进代际（序号不复用），返回新的基线。"""

        value = float(baseline_kg_s)
        if value <= self._tolerance:
            raise ValueError("标定基线必须大于当前流量容差")
        self._baseline = value
        self._next_epoch += 1
        self._epoch = self._next_epoch
        self._active = True
        self._hold_reason = ""
        return self._baseline

    def restore(self, baseline_kg_s: float, epoch: int) -> float:
        """回退到某一次标定：基线与代际一起恢复，不消耗新代际。"""

        value = float(baseline_kg_s)
        if value <= self._tolerance:
            raise ValueError("回退基线必须大于当前流量容差")
        self._baseline = value
        self._epoch = int(epoch)
        self._next_epoch = max(self._next_epoch, self._epoch)
        self._active = True
        self._hold_reason = ""
        return self._baseline

    def note_epoch(self, epoch: int) -> None:
        """把代际计数推进到至少 ``epoch``（恢复台账时用，保证序号不复用）。"""

        self._next_epoch = max(self._next_epoch, int(epoch))

    def holdout(self, reason: str) -> None:
        """停用联锁，等待人工标定；期间判定放行但标明未投入。"""

        self._active = False
        self._hold_reason = str(reason)

    def evaluate(self, flow_kg_s: float) -> dict:
        """判定一次流量是否低于门限；结果携带判定所依据的标定代际。"""

        self._evaluations += 1
        threshold = self.threshold_kg_s()
        margin = float(flow_kg_s) - threshold
        ok = margin >= 0.0 if self._active else True
        if self._active and not ok:
            self._trips += 1
        return {
            "flow_kg_s": round(float(flow_kg_s), 3),
            "threshold_kg_s": round(threshold, 3),
            "margin_kg_s": round(margin, 3),
            "ok": ok,
            "active": self._active,
            "baseline_kg_s": round(self._baseline, 3),
            "tolerance_kg_s": round(self._tolerance, 3),
            "epoch": self._epoch,
        }

    def trips(self) -> int:
        """返回累计触发次数。"""

        return self._trips

    def evaluations(self) -> int:
        """返回累计判定次数。"""

        return self._evaluations
