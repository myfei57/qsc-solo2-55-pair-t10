"""低流量联锁门限与标定代际。

门限不是一个固定值，而是"当前标定基线 - 该代容差"。基线带版本（代际），
标定一次产生一个新一代；换代（启用、回退、停用、重启恢复）一律落
:class:`Migration`，保证联锁判据、标定记录与审计说的是同一次标定。

标定分两阶段，避免一次坏测量直接改动保护：

* ``begin_calibration`` 产生一个 **临时（provisional）** 代并把当前指针切到它，
  联锁立即按它判，但判定结果标明这代尚未确认；上一代用指针留着可回退。
* 确认窗内持续喂实测，``confirm_calibration`` 收尾：实测均值与候选基线偏差
  不超过门限，这代才转 **confirmed** 并开始计有效期；偏差超标这次标定不作数，
  按回路策略 ``rollback``（退回上一代）或 ``disable``（停用联锁等人工标定）。

代际历史只追加（被拒代际也留痕），"当前生效代"用独立指针指向历史中的某一条，
因此回退/停用不会产生重复代际记录。

有效期按时间算（口径定死，不按偏差）：到期门限不作废，只置 ``stale``，联锁
照判但每条判定都带陈旧标记。偏差只用于标定确认，不另设偏差有效期。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import SaltLimits

# 代际状态。
SEED = "seed"
PROVISIONAL = "provisional"
CONFIRMED = "confirmed"
REJECTED = "rejected"
SUPERSEDED = "superseded"

# 换代迁移类型。
MIGRATE_SEED = "seed"
MIGRATE_ACTIVATE = "activate"
MIGRATE_ROLLBACK = "rollback"
MIGRATE_DISABLE = "disable"
MIGRATE_RESTORE = "restore"


@dataclass(frozen=True)
class Generation:
    """一版标定基线及其容差与生命周期状态。"""

    calibration_id: str
    epoch: int
    baseline_kg_s: float
    tolerance_kg_s: float
    measured_kg_s: float
    operator: str
    created_ts: float
    state: str = PROVISIONAL
    confirmed_ts: float | None = None
    expires_ts: float | None = None
    source: str = "calibration"
    reject_reason: str = ""

    @property
    def threshold_kg_s(self) -> float:
        return self.baseline_kg_s - self.tolerance_kg_s

    def is_fresh(self, now: float) -> bool:
        """该代在给定时刻是否仍在有效期内（未到期即新鲜）。"""

        return self.expires_ts is None or now < self.expires_ts

    def as_dict(self) -> dict:
        return {
            "calibration_id": self.calibration_id,
            "epoch": self.epoch,
            "baseline_kg_s": round(self.baseline_kg_s, 3),
            "tolerance_kg_s": round(self.tolerance_kg_s, 3),
            "threshold_kg_s": round(self.threshold_kg_s, 3),
            "measured_kg_s": round(self.measured_kg_s, 3),
            "operator": self.operator,
            "state": self.state,
            "source": self.source,
            "created_ts": round(self.created_ts, 3),
            "confirmed_ts": None if self.confirmed_ts is None else round(self.confirmed_ts, 3),
            "expires_ts": None if self.expires_ts is None else round(self.expires_ts, 3),
            "reject_reason": self.reject_reason,
        }


@dataclass(frozen=True)
class Migration:
    """一次基线换代：启用、回退、停用或重启恢复。"""

    from_epoch: int | None
    to_epoch: int | None
    from_id: str
    to_id: str
    kind: str
    ts: float
    operator: str
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "from_epoch": self.from_epoch,
            "to_epoch": self.to_epoch,
            "from_id": self.from_id,
            "to_id": self.to_id,
            "kind": self.kind,
            "ts": round(self.ts, 3),
            "operator": self.operator,
            "reason": self.reason,
        }


class ThresholdModel:
    """按带版本的标定基线推导低流量联锁门限。"""

    def __init__(self, limits: SaltLimits, baseline_kg_s: float | None = None, now: float = 0.0) -> None:
        self._limits = limits
        self._history: list[Generation] = []
        self._migrations: list[Migration] = []
        self._current = -1  # 指向 _history 中当前生效代际
        self._armed = True
        self._disable_reason = ""
        self._now = float(now)
        self._trips = 0
        self._evaluations = 0
        # 候选代确认窗的实测累积。
        self._confirm_sum = 0.0
        self._confirm_n = 0
        seed_value = float(limits.flow_baseline_kg_s if baseline_kg_s is None else baseline_kg_s)
        self._seed(seed_value, now)

    # ---- 代际管理 -----------------------------------------------------

    def _seed(self, baseline_kg_s: float, now: float) -> None:
        """建立第 0 代出厂基线（不算人工标定，不参与作废回退计数）。"""

        generation = Generation(
            calibration_id="cal-seed",
            epoch=0,
            baseline_kg_s=baseline_kg_s,
            tolerance_kg_s=self._limits.flow_tolerance_kg_s,
            measured_kg_s=baseline_kg_s,
            operator="seed",
            created_ts=float(now),
            state=SEED,
            confirmed_ts=float(now),
            expires_ts=None,
            source="seed",
        )
        self._history.append(generation)
        self._current = len(self._history) - 1
        self._migrations.append(
            Migration(None, 0, "", generation.calibration_id, MIGRATE_SEED, float(now), "seed")
        )

    @property
    def limits(self) -> SaltLimits:
        return self._limits

    @property
    def baseline_kg_s(self) -> float:
        return self.current().baseline_kg_s

    @property
    def tolerance_kg_s(self) -> float:
        return self.current().tolerance_kg_s

    @property
    def epoch(self) -> int:
        """返回当前生效代际。"""

        return self.current().epoch

    @property
    def calibration_id(self) -> str:
        """返回当前判定所依据的标定 id。"""

        return self.current().calibration_id

    def current(self) -> Generation:
        """返回当前生效的代际。"""

        return self._history[self._current]

    def generations(self) -> list[Generation]:
        """返回全部代际历史（含已回退/已拒绝），供记录与控制台展示。"""

        return list(self._history)

    def migrations(self) -> list[Migration]:
        """返回全部换代记录。"""

        return self._migrations

    def armed(self) -> bool:
        """联锁是否投用（作废策略为 disable 且尚未人工标定时为 False）。"""

        return self._armed

    def disable_reason(self) -> str:
        return self._disable_reason

    def has_pending(self) -> bool:
        """当前指针是否指向一个尚未确认的临时代。"""

        return self.current().state == PROVISIONAL

    def threshold_kg_s(self) -> float:
        """返回当前联锁门限（停用时仍给值，但不参与跳闸）。"""

        return self.current().threshold_kg_s

    def calibration_state(self, now: float | None = None) -> str:
        """返回当前代际的标定状态：seed/provisional/confirmed/stale/disabled。"""

        if not self._armed:
            return "disabled"
        generation = self.current()
        if generation.state in (SEED, PROVISIONAL, REJECTED):
            return generation.state
        moment = self._now if now is None else float(now)
        return "stale" if not generation.is_fresh(moment) else CONFIRMED

    def _set_history(self, index: int, generation: Generation) -> None:
        self._history[index] = generation

    # ---- 标定两阶段 ---------------------------------------------------

    def begin_calibration(
        self,
        measured_kg_s: float,
        now: float,
        tolerance_kg_s: float | None = None,
        operator: str = "console",
    ) -> Generation:
        """开始一次标定：产生临时代并立即据此判定，等确认窗收尾。"""

        measured = float(measured_kg_s)
        tol = self._limits.flow_tolerance_kg_s if tolerance_kg_s is None else float(tolerance_kg_s)
        if measured <= tol:
            raise ValueError("实测流量必须大于流量容差")
        if not self._armed:
            raise ValueError("联锁已停用，需先完成一次人工标定恢复")
        if self.has_pending():
            raise ValueError("已有尚未确认的标定，先确认或作废")
        previous = self.current()
        generation = Generation(
            calibration_id=f"cal-{previous.epoch + 1:04d}",
            epoch=previous.epoch + 1,
            baseline_kg_s=measured,
            tolerance_kg_s=tol,
            measured_kg_s=measured,
            operator=str(operator),
            created_ts=float(now),
            state=PROVISIONAL,
            source="calibration",
        )
        self._history.append(generation)
        self._current = len(self._history) - 1
        self._confirm_sum = 0.0
        self._confirm_n = 0
        self._now = float(now)
        return generation

    def observe_confirm(self, flow_kg_s: float) -> None:
        """确认窗内喂一个实测样本。"""

        if self.has_pending():
            self._confirm_sum += float(flow_kg_s)
            self._confirm_n += 1

    def confirm_samples(self) -> int:
        """返回当前候选代确认窗内已喂的实测样本数。"""

        return self._confirm_n

    def confirm_calibration(self, now: float) -> tuple[Generation, Migration | None]:
        """确认窗收尾，返回（生效代际，换代记录）。

        实测样本与候选基线偏差不超标才转 confirmed；超标按回路策略回退或停用。
        """

        if not self.has_pending():
            raise ValueError("没有待确认的标定")
        if self._confirm_n <= 0:
            raise ValueError("确认窗内没有实测样本，无法确认标定")
        candidate_index = self._current
        candidate = self._history[candidate_index]
        previous = self._history[candidate_index - 1]
        observed_mean = self._confirm_sum / self._confirm_n
        deviation = abs(observed_mean - candidate.baseline_kg_s)
        self._now = float(now)

        if deviation <= self._limits.calibration_max_deviation_kg_s:
            confirmed = Generation(
                calibration_id=candidate.calibration_id,
                epoch=candidate.epoch,
                baseline_kg_s=candidate.baseline_kg_s,
                tolerance_kg_s=candidate.tolerance_kg_s,
                measured_kg_s=candidate.measured_kg_s,
                operator=candidate.operator,
                created_ts=candidate.created_ts,
                state=CONFIRMED,
                confirmed_ts=float(now),
                expires_ts=float(now) + self._limits.calibration_valid_for_s,
                source="calibration",
            )
            self._set_history(candidate_index, confirmed)
            self._supersede(previous)
            migration = Migration(
                previous.epoch,
                confirmed.epoch,
                previous.calibration_id,
                confirmed.calibration_id,
                MIGRATE_ACTIVATE,
                float(now),
                confirmed.operator,
                reason=f"deviation={round(deviation, 3)}",
            )
            self._migrations.append(migration)
            self._reset_confirm_window()
            return confirmed, migration

        return self._reject_candidate(candidate_index, f"deviation={round(deviation, 3)} "
                                      f"> max={self._limits.calibration_max_deviation_kg_s}", now)

    def expire_pending(self, now: float) -> tuple[Generation, Migration | None] | None:
        """临时代超过确认窗仍未确认：按作废处理，返回（被拒代际，迁移）。"""

        if not self.has_pending():
            return None
        candidate = self.current()
        if float(now) - candidate.created_ts < self._limits.calibration_confirm_s:
            return None
        return self._reject_candidate(self._current, "confirm_window_expired", now)

    def _supersede(self, generation: Generation) -> None:
        """把上一代确认代标记为被取代（种子代永不改状态）。"""

        for index, item in enumerate(self._history):
            if item is generation and item.state != SEED:
                self._history[index] = _with_state(item, SUPERSEDED)

    def _reject_candidate(
        self, candidate_index: int, reason: str, now: float
    ) -> tuple[Generation, Migration | None]:
        candidate = self._history[candidate_index]
        previous = self._history[candidate_index - 1]
        rejected = Generation(
            calibration_id=candidate.calibration_id,
            epoch=candidate.epoch,
            baseline_kg_s=candidate.baseline_kg_s,
            tolerance_kg_s=candidate.tolerance_kg_s,
            measured_kg_s=candidate.measured_kg_s,
            operator=candidate.operator,
            created_ts=candidate.created_ts,
            state=REJECTED,
            source="calibration",
            reject_reason=reason,
        )
        self._set_history(candidate_index, rejected)
        self._reset_confirm_window()

        if self._limits.calibration_invalid_policy == "disable":
            self._armed = False
            self._disable_reason = reason
            # 当前指针回退上一代；联锁不投用，判定不据此跳闸，只报 disabled。
            self._current = candidate_index - 1
            migration = Migration(
                candidate.epoch,
                None,
                candidate.calibration_id,
                "",
                MIGRATE_DISABLE,
                float(now),
                candidate.operator,
                reason=reason,
            )
            self._migrations.append(migration)
            return rejected, migration

        # rollback：当前指针退回上一代（真实换代迁移 N -> N-1）。
        self._current = candidate_index - 1
        migration = Migration(
            candidate.epoch,
            previous.epoch,
            candidate.calibration_id,
            previous.calibration_id,
            MIGRATE_ROLLBACK,
            float(now),
            candidate.operator,
            reason=reason,
        )
        self._migrations.append(migration)
        return rejected, migration

    def _reset_confirm_window(self) -> None:
        self._confirm_sum = 0.0
        self._confirm_n = 0

    # ---- 有效期与判定 -------------------------------------------------

    def tick(self, now: float) -> None:
        """推进时钟（临时代超时作废由 :meth:`expire_pending` 按调用方条件触发）。"""

        self._now = float(now)

    def evaluate(self, flow_kg_s: float, now: float | None = None) -> dict:
        """判定一次流量是否低于门限，并说清按哪一代标定判。"""

        if now is not None:
            self._now = float(now)
        self._evaluations += 1
        moment = self._now
        generation = self.current()
        threshold = generation.threshold_kg_s
        state = self.calibration_state(moment)
        margin = float(flow_kg_s) - threshold

        if not self._armed:
            # 联锁停用：不跳闸，但必须显式标明未投用，交告警/审计处置。
            ok, armed = True, False
        else:
            armed = True
            ok = margin >= 0.0
            if not ok:
                self._trips += 1

        return {
            "flow_kg_s": round(float(flow_kg_s), 3),
            "threshold_kg_s": round(threshold, 3),
            "margin_kg_s": round(margin, 3),
            "ok": ok,
            "armed": armed,
            "calibration_state": state,
            "calibration_id": generation.calibration_id,
            "baseline_kg_s": round(generation.baseline_kg_s, 3),
            "tolerance_kg_s": round(generation.tolerance_kg_s, 3),
            "epoch": generation.epoch,
            "fresh": state not in ("stale", "disabled"),
            "expires_ts": None if generation.expires_ts is None else round(generation.expires_ts, 3),
        }

    # ---- 恢复与兼容 ---------------------------------------------------

    def calibrate(
        self,
        baseline_kg_s: float,
        tolerance_kg_s: float | None = None,
        now: float = 0.0,
        operator: str = "restore",
    ) -> float:
        """直接落地一代已确认基线（兼容旧入口/恢复用），返回新基线。"""

        value = float(baseline_kg_s)
        tol = self._limits.flow_tolerance_kg_s if tolerance_kg_s is None else float(tolerance_kg_s)
        if value <= tol:
            raise ValueError("标定基线必须大于流量容差")
        previous = self.current()
        generation = Generation(
            calibration_id=f"cal-{previous.epoch + 1:04d}",
            epoch=previous.epoch + 1,
            baseline_kg_s=value,
            tolerance_kg_s=tol,
            measured_kg_s=value,
            operator=str(operator),
            created_ts=float(now),
            state=CONFIRMED,
            confirmed_ts=float(now),
            expires_ts=None,
            source="restore",
        )
        self._history.append(generation)
        self._current = len(self._history) - 1
        self._migrations.append(
            Migration(
                previous.epoch,
                generation.epoch,
                previous.calibration_id,
                generation.calibration_id,
                MIGRATE_RESTORE,
                float(now),
                operator,
            )
        )
        return value

    def restore(
        self,
        calibration_id: str,
        epoch: int,
        baseline_kg_s: float,
        tolerance_kg_s: float,
        state: str,
        created_ts: float,
        confirmed_ts: float | None,
        expires_ts: float | None,
        operator: str = "restore",
        armed: bool = True,
    ) -> Generation:
        """从落盘状态恢复当前代际（跨重启）：历史先只放这一代，记录再追加。"""

        generation = Generation(
            calibration_id=str(calibration_id),
            epoch=int(epoch),
            baseline_kg_s=float(baseline_kg_s),
            tolerance_kg_s=float(tolerance_kg_s),
            measured_kg_s=float(baseline_kg_s),
            operator=str(operator),
            created_ts=float(created_ts),
            state=state if state in (SEED, CONFIRMED, PROVISIONAL) else CONFIRMED,
            confirmed_ts=confirmed_ts,
            expires_ts=expires_ts,
            source="restore",
        )
        self._history = [generation]
        self._current = 0
        self._armed = bool(armed)
        self._now = float(confirmed_ts or created_ts)
        self._migrations.append(
            Migration(
                None,
                generation.epoch,
                "",
                generation.calibration_id,
                MIGRATE_RESTORE,
                float(created_ts),
                operator,
            )
        )
        return generation

    def append_history(self, generation: Generation) -> None:
        """恢复时追加一条代际历史（不改当前指针、不触发换代）。"""

        if not any(item.calibration_id == generation.calibration_id for item in self._history):
            # 当前代已是历史第一条；其余记录按代际顺序追加。
            self._history.append(generation)

    def trips(self) -> int:
        return self._trips

    def evaluations(self) -> int:
        return self._evaluations


def _with_state(generation: Generation, state: str) -> Generation:
    return Generation(
        calibration_id=generation.calibration_id,
        epoch=generation.epoch,
        baseline_kg_s=generation.baseline_kg_s,
        tolerance_kg_s=generation.tolerance_kg_s,
        measured_kg_s=generation.measured_kg_s,
        operator=generation.operator,
        created_ts=generation.created_ts,
        state=state,
        confirmed_ts=generation.confirmed_ts,
        expires_ts=generation.expires_ts,
        source=generation.source,
        reject_reason=generation.reject_reason,
    )
