"""流量标定编排。

流量计读数会随结垢与检修漂移。标定不再"量一下就换基线"，而是走两阶段：

1. :meth:`begin` 产生一个临时基线代，容差取当前实测波动推出的自适应值，联锁
   立即按候选代判但标记未确认，上一代保留可回退；
2. 确认窗内每个控制周期 :meth:`observe` 喂实测，窗口结束 :meth:`confirm`：
   实测均值跟候选基线差太多，这次标定不作数——按回路配置退回上一代
   （``rollback``）或停用联锁等人工标定（``disable``）。

标定记录（每代基线）、换代迁移与审计共用同一套 ``calibration_id`` / ``epoch``，
每条联锁判定都能说清是按哪一次标定判的。
"""

from __future__ import annotations

from ..config import SaltLimits
from ..receiver.threshold import Generation, Migration, ThresholdModel
from .flowstats import FlowStats


class FlowCalibrator:
    """把现场实测流量写回联锁基线，并管理标定代际生命周期。"""

    def __init__(self, limits: SaltLimits, threshold: ThresholdModel) -> None:
        self._limits = limits
        self._threshold = threshold
        self._stats = FlowStats(limits)

    @property
    def limits(self) -> SaltLimits:
        return self._limits

    @property
    def threshold(self) -> ThresholdModel:
        return self._threshold

    @property
    def stats(self) -> FlowStats:
        return self._stats

    # ---- 运行期观测 ---------------------------------------------------

    def observe(self, flow_kg_s: float, running: bool, now: float) -> dict | None:
        """每个控制周期喂一次：在运时累计波动，并推进候选标定确认窗。

        返回本周期发生的标定状态变化（确认通过 / 回退 / 停用 / 超时作废），
        没有变化时返回 ``None``。
        """

        change: dict | None = None
        if running:
            self._stats.observe(flow_kg_s)
        # 推进时钟；临时代到点且仍无在运实测才按超时作废，有样本则走偏差确认。
        if self._threshold.has_pending():
            self._threshold.tick(now)
            if running:
                self._threshold.observe_confirm(flow_kg_s)
            candidate = self._threshold.current()
            window_elapsed = now - candidate.created_ts >= self._limits.calibration_confirm_s
            if window_elapsed:
                if self._threshold.confirm_samples() > 0:
                    generation, migration = self._threshold.confirm_calibration(now)
                    change = self._change(generation, migration)
                else:
                    expired = self._threshold.expire_pending(now)
                    if expired is not None:
                        rejected, migration = expired
                        change = self._change(rejected, migration)
        return change

    def _change(self, generation: Generation, migration: Migration | None) -> dict:
        return {
            "calibration_id": generation.calibration_id,
            "epoch": generation.epoch,
            "state": generation.state,
            "reject_reason": generation.reject_reason,
            "migration": None if migration is None else migration.as_dict(),
        }

    # ---- 标定两阶段 ---------------------------------------------------

    def begin(self, measured_kg_s: float, now: float, operator: str = "console") -> Generation:
        """开始一次标定：容差取实测波动自适应值，产生临时代。"""

        tolerance = self._stats.tolerance_kg_s()
        return self._threshold.begin_calibration(measured_kg_s, now, tolerance, operator)

    def confirm(self, now: float) -> tuple[Generation, Migration | None]:
        """确认窗收尾：偏差合格转 confirmed，否则按回路策略作废。"""

        return self._threshold.confirm_calibration(now)

    def calibrate(self, measured_kg_s: float, now: float, operator: str = "console") -> Generation:
        """兼容旧入口：开始标定。正式生效要走确认窗（见 :meth:`observe`）。"""

        return self.begin(measured_kg_s, now, operator)

    # ---- 查询 ---------------------------------------------------------

    def records(self) -> list[Generation]:
        """返回全部标定代际记录。"""

        return self._threshold.generations()

    def migrations(self) -> list[Migration]:
        """返回全部换代迁移记录。"""

        return self._threshold.migrations()

    def last(self) -> Generation | None:
        return self._threshold.generations()[-1] if self._threshold.generations() else None

    def active(self) -> Generation:
        """返回当前生效代际。"""

        return self._threshold.current()

    def pending(self) -> Generation | None:
        return self._threshold.current() if self._threshold.has_pending() else None

    def baseline_kg_s(self) -> float:
        return self._threshold.baseline_kg_s

    def threshold_kg_s(self) -> float:
        return self._threshold.threshold_kg_s

    def tolerance_kg_s(self) -> float:
        return self._threshold.tolerance_kg_s
