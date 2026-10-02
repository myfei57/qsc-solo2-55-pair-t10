"""盐侧持久化。

冷热罐液位、流量基线与标定代际跨重启必须保留：重启后如果液位回到初值，操作员
看到的存量就是假的；基线丢失则联锁门限回到出厂值，低流量保护会误动。

标定记录（每一代基线）、换代迁移与当前生效代际分开落盘：记录只追加供控制台
展示与审计回放，当前代际单独一份供启动时恢复门限。
"""

from __future__ import annotations

from ..store.jsonfile import JsonStore

LEVELS_DOC = "salt/levels"
BASELINE_DOC = "salt/baseline"
CALIBRATIONS_DOC = "salt/calibrations"
MIGRATIONS_DOC = "salt/migrations"


class SaltPersistence:
    """盐侧状态读写。"""

    def __init__(self, store: JsonStore) -> None:
        self._store = store

    @property
    def store(self) -> JsonStore:
        return self._store

    def save_levels(self, cold_pct: float, hot_pct: float, ts: float) -> str:
        """落盘冷热罐液位。"""

        return self._store.write(
            LEVELS_DOC,
            {
                "cold_pct": round(float(cold_pct), 4),
                "hot_pct": round(float(hot_pct), 4),
                "ts": round(float(ts), 3),
            },
        )

    def read_levels(self) -> dict | None:
        """读回冷热罐液位。"""

        raw = self._store.read(LEVELS_DOC)
        if not isinstance(raw, dict):
            return None
        cold = raw.get("cold_pct")
        hot = raw.get("hot_pct")
        if not isinstance(cold, (int, float)) or not isinstance(hot, (int, float)):
            return None
        return {"cold_pct": float(cold), "hot_pct": float(hot), "ts": float(raw.get("ts", 0.0))}

    def save_baseline(self, baseline_kg_s: float, epoch: int) -> str:
        """落盘流量基线与代际（兼容旧口径，新代码请用 save_active_generation）。"""

        return self._store.write(
            BASELINE_DOC,
            {"baseline_kg_s": round(float(baseline_kg_s), 4), "epoch": int(epoch)},
        )

    def read_baseline(self) -> dict | None:
        """读回流量基线。"""

        raw = self._store.read(BASELINE_DOC)
        if not isinstance(raw, dict):
            return None
        value = raw.get("baseline_kg_s")
        if not isinstance(value, (int, float)):
            return None
        return {"baseline_kg_s": float(value), "epoch": int(raw.get("epoch", 0))}

    def save_active_generation(self, generation: dict, armed: bool) -> str:
        """落盘当前生效代际，供重启恢复门限与有效期。"""

        return self._store.write(
            BASELINE_DOC,
            {
                "baseline_kg_s": round(float(generation["baseline_kg_s"]), 4),
                "tolerance_kg_s": round(float(generation["tolerance_kg_s"]), 4),
                "threshold_kg_s": round(float(generation["threshold_kg_s"]), 4),
                "epoch": int(generation["epoch"]),
                "calibration_id": str(generation["calibration_id"]),
                "state": str(generation["state"]),
                "created_ts": float(generation["created_ts"]),
                "confirmed_ts": generation.get("confirmed_ts"),
                "expires_ts": generation.get("expires_ts"),
                "operator": str(generation.get("operator", "restore")),
                "armed": bool(armed),
            },
        )

    def read_active_generation(self) -> dict | None:
        """读回当前生效代际；旧版只有基线的落盘也能读。"""

        raw = self._store.read(BASELINE_DOC)
        if not isinstance(raw, dict):
            return None
        value = raw.get("baseline_kg_s")
        if not isinstance(value, (int, float)):
            return None
        return {
            "baseline_kg_s": float(value),
            "tolerance_kg_s": float(raw.get("tolerance_kg_s", 0.0)) or None,
            "epoch": int(raw.get("epoch", 0)),
            "calibration_id": raw.get("calibration_id"),
            "state": raw.get("state"),
            "created_ts": float(raw.get("created_ts", 0.0)),
            "confirmed_ts": raw.get("confirmed_ts"),
            "expires_ts": raw.get("expires_ts"),
            "operator": str(raw.get("operator", "restore")),
            "armed": bool(raw.get("armed", True)),
        }

    def save_calibrations(self, generations: list[dict]) -> str:
        """落盘全部标定代际记录（只追加语义，整体覆盖原子写）。"""

        return self._store.write(CALIBRATIONS_DOC, {"generations": generations})

    def read_calibrations(self) -> list[dict]:
        """读回标定代际记录。"""

        raw = self._store.read(CALIBRATIONS_DOC)
        if not isinstance(raw, dict) or not isinstance(raw.get("generations"), list):
            return []
        return [item for item in raw["generations"] if isinstance(item, dict)]

    def save_migrations(self, migrations: list[dict]) -> str:
        """落盘全部换代迁移记录。"""

        return self._store.write(MIGRATIONS_DOC, {"migrations": migrations})

    def read_migrations(self) -> list[dict]:
        """读回换代迁移记录。"""

        raw = self._store.read(MIGRATIONS_DOC)
        if not isinstance(raw, dict) or not isinstance(raw.get("migrations"), list):
            return []
        return [item for item in raw["migrations"] if isinstance(item, dict)]

    def documents(self) -> list[str]:
        """返回盐侧已落盘的文档列表。"""

        tracked = (LEVELS_DOC, BASELINE_DOC, CALIBRATIONS_DOC, MIGRATIONS_DOC)
        return [doc for doc in self._store.list_ids() if doc in tracked]
