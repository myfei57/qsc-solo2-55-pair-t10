"""自适应容差、标定两阶段、有效期、回退/停用、换代迁移与持久化。"""

from __future__ import annotations

import dataclasses

import pytest

from cspplant.app import build_plant
from cspplant.config import default_config
from cspplant.errors import InterlockError
from cspplant.receiver.threshold import (
    CONFIRMED,
    PROVISIONAL,
    REJECTED,
    SEED,
    ThresholdModel,
)
from cspplant.salt.flowstats import FlowStats


def _salt(**overrides):
    return dataclasses.replace(default_config().salt, **overrides)


def _config(**salt_overrides):
    cfg = default_config()
    return dataclasses.replace(cfg, salt=_salt(**salt_overrides))


# ---- 自适应容差 -------------------------------------------------------


def test_adaptive_tolerance_tracks_observed_sigma():
    limits = _salt(tolerance_k_sigma=3.0, tolerance_floor_kg_s=2.0, tolerance_cap_kg_s=18.0)
    stats = FlowStats(limits)
    # 样本不足时回退种子容差。
    assert stats.tolerance_kg_s() == pytest.approx(limits.flow_tolerance_kg_s)
    for _ in range(40):
        stats.observe(58.0)
        stats.observe(60.0)  # 稳态交替 -> sigma 约 1
    sigma = stats.sigma_kg_s()
    assert sigma is not None and sigma > 0.5
    assert stats.tolerance_kg_s() == pytest.approx(3.0 * sigma, rel=0.2)


def test_adaptive_tolerance_is_clamped_to_floor_and_cap():
    limits = _salt(tolerance_k_sigma=3.0, tolerance_floor_kg_s=4.0, tolerance_cap_kg_s=10.0)
    assert limits.adaptive_tolerance_kg_s(0.01) == pytest.approx(4.0)
    assert limits.adaptive_tolerance_kg_s(100.0) == pytest.approx(10.0)
    assert limits.adaptive_tolerance_kg_s(2.0) == pytest.approx(6.0)


# ---- 两阶段标定 -------------------------------------------------------


def test_calibration_is_provisional_until_confirmed():
    model = ThresholdModel(_salt(calibration_confirm_s=10.0, calibration_max_deviation_kg_s=5.0), now=0.0)
    assert model.calibration_state(0.0) == SEED
    model.begin_calibration(70.0, 0.0, 9.0, "op")
    assert model.has_pending() is True
    assert model.calibration_state(0.0) == PROVISIONAL
    assert model.epoch == 1
    # 临时代立即参与判定。
    assert model.threshold_kg_s() == pytest.approx(61.0)
    # 确认窗实测与候选基线吻合 -> 生效。
    for _ in range(5):
        model.observe_confirm(70.5)
    generation, migration = model.confirm_calibration(10.0)
    assert generation.state == CONFIRMED
    assert model.calibration_state(10.0) == CONFIRMED
    assert migration.kind == "activate"
    assert (migration.from_epoch, migration.to_epoch) == (0, 1)
    # 有效期开始计时。
    assert generation.expires_ts == pytest.approx(10.0 + default_config().salt.calibration_valid_for_s)


def test_calibration_requires_observation_samples():
    model = ThresholdModel(_salt(), now=0.0)
    model.begin_calibration(70.0, 0.0, 9.0)
    with pytest.raises(ValueError):
        model.confirm_calibration(1.0)


def test_cannot_begin_second_calibration_while_pending():
    model = ThresholdModel(_salt(), now=0.0)
    model.begin_calibration(70.0, 0.0, 9.0)
    with pytest.raises(ValueError):
        model.begin_calibration(72.0, 1.0, 9.0)


# ---- 作废：回退 / 停用（按回路）---------------------------------------


def test_rejected_calibration_rolls_back_to_previous_generation():
    model = ThresholdModel(
        _salt(calibration_max_deviation_kg_s=3.0, calibration_invalid_policy="rollback"), now=0.0
    )
    seed_threshold = model.threshold_kg_s()
    model.begin_calibration(80.0, 0.0, 9.0)
    for _ in range(5):
        model.observe_confirm(58.0)  # 与候选 80 相差 22
    rejected, migration = model.confirm_calibration(10.0)
    assert rejected.state == REJECTED
    assert "deviation" in rejected.reject_reason
    # 当前判定退回第 0 代。
    assert model.epoch == 0
    assert model.calibration_id == "cal-seed"
    assert model.threshold_kg_s() == pytest.approx(seed_threshold)
    assert model.armed() is True
    assert migration.kind == "rollback"
    assert (migration.from_epoch, migration.to_epoch) == (1, 0)
    # 被拒代际仍保留在记录里。
    assert any(g.state == REJECTED and g.epoch == 1 for g in model.generations())


def test_rejected_calibration_disables_interlock_per_loop_policy():
    model = ThresholdModel(
        _salt(calibration_max_deviation_kg_s=3.0, calibration_invalid_policy="disable"), now=0.0
    )
    model.begin_calibration(80.0, 0.0, 9.0)
    for _ in range(5):
        model.observe_confirm(58.0)
    rejected, migration = model.confirm_calibration(10.0)
    assert rejected.state == REJECTED
    assert model.armed() is False
    assert model.calibration_state(10.0) == "disabled"
    assert migration.kind == "disable"
    # 停用后判定不跳闸，但显式标 armed=False。
    verdict = model.evaluate(10.0, 10.0)
    assert verdict["ok"] is True
    assert verdict["armed"] is False
    # 联锁停用时不允许再标定（必须先人工恢复路径）。
    with pytest.raises(ValueError):
        model.begin_calibration(60.0, 11.0, 9.0)


def test_pending_calibration_expiring_confirmation_window_is_rejected():
    model = ThresholdModel(
        _salt(calibration_confirm_s=10.0, calibration_invalid_policy="rollback"), now=0.0
    )
    model.begin_calibration(80.0, 0.0, 9.0)
    assert model.expire_pending(5.0) is None  # 窗未到
    rejected, migration = model.expire_pending(11.0)
    assert rejected.state == REJECTED
    assert rejected.reject_reason == "confirm_window_expired"
    assert model.epoch == 0


# ---- 有效期按时间算（定死口径）----------------------------------------


def test_expired_calibration_is_stale_but_still_judges():
    model = ThresholdModel(
        _salt(calibration_valid_for_s=100.0, calibration_max_deviation_kg_s=5.0), now=0.0
    )
    model.begin_calibration(70.0, 0.0, 9.0)
    for _ in range(5):
        model.observe_confirm(70.0)
    model.confirm_calibration(10.0)  # 到期时刻 = 110
    assert model.calibration_state(50.0) == CONFIRMED
    assert model.calibration_state(110.0) == "stale"
    # 陈旧不清空门限，仍按末代基线判；低流量照样触发。
    verdict = model.evaluate(30.0, 110.0)
    assert verdict["ok"] is False
    assert verdict["calibration_state"] == "stale"
    assert verdict["fresh"] is False
    assert verdict["calibration_id"] == "cal-0001"


def test_every_verdict_carries_calibration_identity():
    model = ThresholdModel(_salt(), now=0.0)
    verdict = model.evaluate(58.0, 0.0)
    assert verdict["calibration_id"] == "cal-seed"
    assert verdict["epoch"] == 0
    assert "baseline_kg_s" in verdict and "tolerance_kg_s" in verdict


# ---- 平台端到端 -------------------------------------------------------


def _bring_online(platform):
    platform.start()
    platform.track(800.0, 720.0)
    platform.circulate()
    platform.dispatch(100.0, "fixture")


def test_platform_calibration_confirms_after_window(platform, clock):
    _bring_online(platform)
    for _ in range(35):
        platform.cycle(1.0)
    record = platform.calibrate_flow(60.0, "op")
    assert record["state"] == PROVISIONAL
    for _ in range(320):
        clock.advance(1.0)
        platform.cycle(1.0)
    salt = platform.status()["salt"]
    assert salt["calibration_state"] == "stale" or salt["epoch"] == 1
    assert salt["epoch"] == 1
    assert any(m["kind"] == "activate" for m in salt["migrations"])


def _platform(tmp_path, clock, **salt_overrides):
    cfg = _config(**salt_overrides) if salt_overrides else default_config()
    return build_plant(cfg, clock=clock, storage_root=str(tmp_path / "state"))


def test_platform_rollback_when_measured_deviation_too_large(tmp_path, clock):
    platform = _platform(
        tmp_path, clock,
        calibration_confirm_s=10.0,
        calibration_max_deviation_kg_s=3.0,
    )
    _bring_online(platform)
    for _ in range(35):
        platform.cycle(1.0)
    before_epoch = platform.salt.threshold.epoch
    platform.calibrate_flow(95.0, "op")  # 实际流量约 58
    for _ in range(12):
        clock.advance(1.0)
        platform.cycle(1.0)
    salt = platform.status()["salt"]
    assert salt["epoch"] == before_epoch
    assert salt["interlock_armed"] is True
    assert any(g["state"] == REJECTED for g in salt["calibrations"])
    assert any(m["kind"] == "rollback" for m in salt["migrations"])
    actions = [entry["action"] for entry in platform.audit.entries(60)]
    assert "flow_calibration_rejected" in actions


def test_platform_disable_policy_alarm_and_audit(tmp_path, clock):
    platform = _platform(
        tmp_path, clock,
        calibration_invalid_policy="disable",
        calibration_confirm_s=10.0,
        calibration_max_deviation_kg_s=3.0,
    )
    _bring_online(platform)
    for _ in range(35):
        platform.cycle(1.0)
    platform.calibrate_flow(95.0, "op")
    for _ in range(12):
        clock.advance(1.0)
        platform.cycle(1.0)
    assert platform.salt.threshold.armed() is False
    actions = [entry["action"] for entry in platform.audit.entries(80)]
    assert "flow_calibration_disabled" in actions


def test_calibration_records_persist_across_restart(platform, clock):
    _bring_online(platform)
    for _ in range(35):
        platform.cycle(1.0)
    platform.calibrate_flow(60.0, "op")
    for _ in range(320):
        clock.advance(1.0)
        platform.cycle(1.0)
    before = platform.status()["salt"]
    assert before["epoch"] == 1
    tolerance_before = before["tolerance_kg_s"]

    restarted = build_plant(
        platform.config, clock=clock, storage_root=platform.store.root
    )
    after = restarted.status()["salt"]
    assert after["epoch"] == 1
    assert after["calibration_id"] == before["calibration_id"]
    assert after["tolerance_kg_s"] == pytest.approx(tolerance_before)
    assert len(after["calibrations"]) == len(before["calibrations"])
    kinds = [m["kind"] for m in after["migrations"]]
    assert "activate" in kinds


def test_console_api_exposes_calibration_records(platform):
    from cspplant.console.api import Api

    _bring_online(platform)
    api = Api(platform)
    status, payload = api.handle("GET", "/api/calibrations", {}, {})
    assert status == 200
    assert payload["current"]["calibration_id"] == "cal-seed"
    assert any(g["calibration_id"] == "cal-seed" for g in payload["calibrations"])
    assert payload["interlock_armed"] is True

    # 开始标定后接口能看到临时代。
    status, payload = api.handle("POST", "/api/calibrate", {}, {"measured_kg_s": 60.0})
    assert status == 200 and payload["state"] == PROVISIONAL
    status, payload = api.handle("GET", "/api/calibrations", {}, {})
    assert payload["current"]["state"] == PROVISIONAL


def test_salt_page_lists_calibration_generations(platform):
    from cspplant.console.pages import render_page

    _bring_online(platform)
    platform.calibrate_flow(60.0, "op")
    html = render_page("/salt", platform)
    assert "cal-0001" in html
    assert "待确认" in html
