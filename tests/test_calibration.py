"""流量标定：自适应容差、偏差有效期、作废回退／停用与迁移。"""

from __future__ import annotations

import dataclasses

import pytest

from cspplant.app import build_plant
from cspplant.config import INVALIDATION_HOLDOUT, default_config
from cspplant.errors import ConfigError, StorageError
from cspplant.receiver.threshold import ThresholdModel
from cspplant.salt.calibrate import FlowCalibrator
from cspplant.config import SaltLimits


def _limits(**overrides) -> SaltLimits:
    return dataclasses.replace(default_config().salt, **overrides)


def _calibrator(**overrides) -> tuple[FlowCalibrator, ThresholdModel]:
    limits = _limits(**overrides)
    threshold = ThresholdModel(limits)
    return FlowCalibrator(limits, threshold), threshold


def _feed(calibrator: FlowCalibrator, values: list[float], dt: float = 1.0) -> list[dict]:
    events: list[dict] = []
    for value in values:
        events.extend(calibrator.observe(value, dt))
    return events


# ---- 自适应容差 -------------------------------------------------------


def test_tolerance_stays_at_factory_until_enough_samples():
    calibrator, threshold = _calibrator()
    _feed(calibrator, [56.0, 60.0] * 14)  # 28 个样本，不足 30
    assert threshold.tolerance_kg_s == pytest.approx(9.0)
    _feed(calibrator, [56.0, 60.0])  # 第 29、30 个样本，开始自适应
    assert threshold.tolerance_kg_s != pytest.approx(9.0)


def test_tolerance_tightens_when_flow_is_quiet():
    calibrator, threshold = _calibrator()
    _feed(calibrator, [56.0, 60.0] * 20)  # σ ≈ 2.0 → 容差 ≈ 3σ = 6.1 < 出厂 9
    assert 5.5 < threshold.tolerance_kg_s < 6.6
    assert threshold.threshold_kg_s() == pytest.approx(58.0 - threshold.tolerance_kg_s)


def test_tolerance_loosens_when_flow_is_noisy():
    calibrator, threshold = _calibrator()
    _feed(calibrator, [54.0, 62.0] * 20)  # σ ≈ 4.1 → 容差 ≈ 12.3 > 出厂 9
    assert threshold.tolerance_kg_s > 11.0


def test_tolerance_is_clamped_to_configured_band():
    calibrator, threshold = _calibrator()
    _feed(calibrator, [58.0] * 40)  # 无波动 → 夹到下限
    assert threshold.tolerance_kg_s == pytest.approx(4.0)
    noisy, noisy_threshold = _calibrator()
    _feed(noisy, [44.0, 72.0] * 20)  # 剧烈波动 → 夹到上限
    assert noisy_threshold.tolerance_kg_s == pytest.approx(14.0)
    assert noisy_threshold.threshold_kg_s() > 0.0


# ---- 有效期（按实测偏差，不按时间） -----------------------------------


def test_deviation_must_persist_before_voiding():
    calibrator, threshold = _calibrator()
    calibrator.calibrate(80.0, now=100.0, operator="test")
    # 29 秒越带：不到确认窗口，不作废。
    assert _feed(calibrator, [30.0] * 29) == []
    assert threshold.epoch == 1
    # 回到带内：累计清零。
    assert _feed(calibrator, [79.0]) == []
    # 重新越带又要从头累计。
    assert _feed(calibrator, [30.0] * 29) == []
    events = _feed(calibrator, [30.0])
    assert [event["action"] for event in events] == [
        "flow_calibration_voided",
        "flow_calibration_rollback",
    ]


def test_voided_calibration_rolls_back_to_previous_valid_one():
    calibrator, threshold = _calibrator()
    calibrator.calibrate(70.0, now=100.0, operator="a")
    calibrator.calibrate(80.0, now=200.0, operator="b")
    events = _feed(calibrator, [30.0] * 30)
    assert events[0]["epoch"] == 2  # 作废的是第二次标定
    assert events[1]["action"] == "flow_calibration_rollback"
    assert threshold.epoch == 1
    assert threshold.baseline_kg_s == pytest.approx(70.0)
    records = calibrator.records()
    assert records[1].voided is True and "偏差" in records[1].void_reason
    assert records[0].voided is False
    # 联锁判定引用回退后的代际。
    verdict = threshold.evaluate(70.0)
    assert verdict["epoch"] == 1
    assert verdict["baseline_kg_s"] == pytest.approx(70.0)


def test_rollback_falls_to_factory_baseline_then_holdout():
    calibrator, threshold = _calibrator()
    calibrator.calibrate(80.0, now=100.0, operator="a")
    _feed(calibrator, [30.0] * 30)  # 作废代际 1 → 退回出厂基线（代际 0）
    assert threshold.epoch == 0
    assert threshold.baseline_kg_s == pytest.approx(58.0)
    assert threshold.active is True
    events = _feed(calibrator, [30.0] * 30)  # 出厂基线也被作废 → 停用等人工
    assert [event["action"] for event in events] == [
        "flow_calibration_voided",
        "flow_interlock_holdout",
    ]
    assert threshold.active is False
    verdict = threshold.evaluate(1.0)
    assert verdict["ok"] is True and verdict["active"] is False
    # 人工重新标定后联锁恢复投入，且代际不复用。
    calibrator.calibrate(60.0, now=300.0, operator="manual")
    assert threshold.active is True
    assert threshold.epoch == 2


def test_holdout_policy_disables_interlock_immediately():
    calibrator, threshold = _calibrator(interlock_on_invalidation=INVALIDATION_HOLDOUT)
    calibrator.calibrate(80.0, now=100.0, operator="a")
    events = _feed(calibrator, [30.0] * 30)
    assert [event["action"] for event in events] == [
        "flow_calibration_voided",
        "flow_interlock_holdout",
    ]
    assert threshold.active is False
    assert threshold.hold_reason != ""
    assert threshold.evaluate(1.0)["ok"] is True  # 停用期间联锁不动作


def test_epochs_are_never_reused_after_rollback():
    calibrator, threshold = _calibrator()
    calibrator.calibrate(70.0, now=100.0, operator="a")
    calibrator.calibrate(80.0, now=200.0, operator="b")
    _feed(calibrator, [30.0] * 30)  # 作废代际 2，回退到代际 1
    assert threshold.epoch == 1
    calibrator.calibrate(75.0, now=300.0, operator="c")
    assert threshold.epoch == 3  # 新标定取新序号，不复用 2


# ---- 配置校验 ---------------------------------------------------------


def test_config_rejects_unknown_invalidation_policy():
    from cspplant.config import SaltLimits as Limits

    section = {
        "cold_level_min_pct": 18.0,
        "hot_level_max_pct": 96.0,
        "pump_ramp_kg_s2": 1.4,
        "handoff_confirm_s": 15.0,
        "flow_baseline_kg_s": 58.0,
        "flow_tolerance_kg_s": 9.0,
        "flow_tolerance_min_kg_s": 4.0,
        "flow_tolerance_max_kg_s": 14.0,
        "flow_tolerance_sigma_mult": 3.0,
        "calibration_min_samples": 30,
        "calibration_max_deviation_kg_s": 18.0,
        "calibration_confirm_s": 30.0,
        "interlock_on_invalidation": "guess",
        "header_pressure_target_mpa": 2.5,
        "header_pressure_max_mpa": 3.4,
    }
    with pytest.raises(ConfigError):
        Limits.from_mapping(section)
    section["interlock_on_invalidation"] = "rollback"
    section["calibration_max_deviation_kg_s"] = 10.0  # 窄于容差上限 14
    with pytest.raises(ConfigError):
        Limits.from_mapping(section)


# ---- 平台级：回路、审计与控制台 ---------------------------------------


def _ramp_cold_pump(platform) -> None:
    platform.salt.start_cold()
    for _ in range(60):
        platform.salt.step(1.0)


def test_loop_step_voids_bad_calibration_and_audits(platform):
    platform.start()
    _ramp_cold_pump(platform)
    first = platform.calibrate_flow(72.0, "shift-lead")
    second = platform.calibrate_flow(80.0, "shift-lead")
    assert (first["epoch"], second["epoch"]) == (1, 2)
    for _ in range(29):
        platform.cycle(1.0)
    assert platform.salt.threshold.epoch == 2  # 确认窗口未满，仍按第二次标定
    platform.cycle(1.0)
    # 偏差持续越带：第二次标定作废，回退到第一次。
    assert platform.salt.threshold.epoch == 1
    assert platform.salt.threshold.baseline_kg_s == pytest.approx(72.0)
    records = platform.salt.calibrator.records()
    assert records[1].voided is True
    assert records[0].voided is False

    entries = platform.audit.entries(limit=50)
    by_action = {}
    for entry in entries:
        by_action.setdefault(entry["action"], []).append(entry)
    # 联锁判据、标定记录与审计口径一致：同一代际、同一基线、同一容差。
    recalibrated = by_action["flow_recalibrated"]
    assert [item["payload"]["epoch"] for item in recalibrated] == [2, 1]  # 最新在前
    assert recalibrated[0]["payload"]["tolerance_kg_s"] == pytest.approx(second["tolerance_kg_s"])
    voided = by_action["flow_calibration_voided"][0]
    assert voided["severity"] == "warning"
    assert voided["payload"]["epoch"] == 2
    rollback = by_action["flow_calibration_rollback"][0]
    assert rollback["payload"]["epoch"] == 1
    assert rollback["payload"]["baseline_kg_s"] == pytest.approx(72.0)
    # 当前联锁判定引用的就是回退后的那次标定。
    verdict = platform.receiver.threshold.evaluate(72.0)
    assert verdict["epoch"] == 1
    assert verdict["tolerance_kg_s"] == pytest.approx(platform.salt.threshold.tolerance_kg_s)


def test_holdout_policy_platform_wide(platform, clock):
    import dataclasses as dc

    config = dc.replace(
        platform.config,
        salt=dc.replace(platform.config.salt, interlock_on_invalidation="holdout"),
    )
    plant = build_plant(config, clock=clock, storage_root=platform.store.root + "-holdout")
    plant.start()
    plant.salt.start_cold()
    for _ in range(60):
        plant.salt.step(1.0)
    plant.calibrate_flow(80.0, "shift-lead")
    for _ in range(30):
        plant.cycle(1.0)
    assert plant.salt.threshold.active is False
    verdict = plant.receiver.threshold.evaluate(1.0)
    assert verdict["ok"] is True and verdict["active"] is False
    entries = plant.audit.entries(limit=50)
    holdout = [entry for entry in entries if entry["action"] == "flow_interlock_holdout"]
    assert holdout and holdout[0]["severity"] == "alarm"


def test_console_exposes_calibration_ledger(platform):
    from cspplant.console.api import Api
    from cspplant.console.pages import render_page

    platform.start()
    platform.calibrate_flow(72.0, "shift-lead")
    api = Api(platform)
    status, payload = api.handle("GET", "/api/calibrations", {}, {})
    assert status == 200
    assert payload["epoch"] == 1
    assert payload["interlock_active"] is True
    assert payload["records"][0]["operator"] == "shift-lead"
    assert payload["records"][0]["voided"] is False
    html = render_page("/salt", platform)
    assert "标定记录" in html and "自适应容差" in html


# ---- 持久化与迁移 -----------------------------------------------------


def test_baseline_v1_document_migrates_on_read(tmp_path, clock):
    from cspplant.store.jsonfile import JsonStore

    root = str(tmp_path / "state")
    store = JsonStore(root)
    store.write("salt/baseline", {"baseline_kg_s": 62.0, "epoch": 3})  # v1：无版本号
    plant = build_plant(default_config(), clock=clock, storage_root=root)
    assert plant.salt.threshold.baseline_kg_s == pytest.approx(62.0)
    assert plant.salt.threshold.epoch == 3
    assert plant.salt.threshold.tolerance_kg_s == pytest.approx(9.0)  # 迁移后回退出厂容差
    assert plant.salt.threshold.active is True


def test_baseline_newer_version_is_rejected(tmp_path, clock):
    from cspplant.store.jsonfile import JsonStore

    root = str(tmp_path / "state")
    store = JsonStore(root)
    store.write("salt/baseline", {"version": 99, "baseline_kg_s": 62.0, "epoch": 3})
    with pytest.raises(StorageError):
        build_plant(default_config(), clock=clock, storage_root=root)


def test_ledger_and_holdout_survive_restart(platform, clock):
    platform.start()
    _ramp_cold_pump(platform)
    platform.calibrate_flow(72.0, "shift-lead")
    platform.calibrate_flow(80.0, "shift-lead")
    for _ in range(30):
        platform.cycle(1.0)  # 代际 2 作废，回退到代际 1

    restarted = build_plant(platform.config, clock=clock, storage_root=platform.store.root)
    assert restarted.salt.threshold.epoch == 1
    assert restarted.salt.threshold.baseline_kg_s == pytest.approx(72.0)
    records = restarted.salt.calibrator.records()
    assert len(records) == 2
    assert records[1].voided is True
    # 代际序号接续，不复用。
    record = restarted.calibrate_flow(75.0, "second-shift")
    assert record["epoch"] == 3


def test_holdout_state_survives_restart(tmp_path, clock):
    import dataclasses as dc

    config = dc.replace(
        default_config(),
        salt=dc.replace(default_config().salt, interlock_on_invalidation="holdout"),
    )
    root = str(tmp_path / "state")
    plant = build_plant(config, clock=clock, storage_root=root)
    plant.start()
    plant.salt.start_cold()
    for _ in range(60):
        plant.salt.step(1.0)
    plant.calibrate_flow(80.0, "shift-lead")
    for _ in range(30):
        plant.cycle(1.0)
    assert plant.salt.threshold.active is False

    restarted = build_plant(config, clock=clock, storage_root=root)
    assert restarted.salt.threshold.active is False
    assert restarted.salt.threshold.hold_reason != ""
