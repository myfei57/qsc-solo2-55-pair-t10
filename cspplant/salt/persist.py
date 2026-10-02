"""盐侧持久化。

冷热罐液位与流量基线跨重启必须保留：重启后如果液位回到初值，操作员看到的
存量就是假的；基线丢失则联锁门限回到出厂值，低流量保护会误动。标定台账也
一并落盘，否则重启后"上一次标定"无从查起，作废回退没有目标。

基线文档带版本号，换代属于迁移：读出时按版本号逐代升级到当前结构，老文档
不丢弃、不硬改；遇到比当前还新的版本号直接报 :class:`StorageError`，不猜。
"""

from __future__ import annotations

from ..errors import StorageError
from ..store.jsonfile import JsonStore

LEVELS_DOC = "salt/levels"
BASELINE_DOC = "salt/baseline"
CALIBRATIONS_DOC = "salt/calibrations"

# 基线文档当前版本：v1 只有 baseline_kg_s/epoch；v2 补容差、投入状态与时间戳。
BASELINE_VERSION = 2
CALIBRATIONS_VERSION = 1
# 台账只保留最近若干次标定，避免无限增长。
LEDGER_LIMIT = 50


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

    def save_baseline(
        self,
        baseline_kg_s: float,
        epoch: int,
        tolerance_kg_s: float,
        active: bool,
        hold_reason: str = "",
        ts: float = 0.0,
    ) -> str:
        """落盘流量基线、代际、容差与联锁投入状态（当前版本）。"""

        return self._store.write(
            BASELINE_DOC,
            {
                "version": BASELINE_VERSION,
                "baseline_kg_s": round(float(baseline_kg_s), 4),
                "epoch": int(epoch),
                "tolerance_kg_s": round(float(tolerance_kg_s), 4),
                "active": bool(active),
                "hold_reason": str(hold_reason),
                "ts": round(float(ts), 3),
            },
        )

    def read_baseline(self) -> dict | None:
        """读回流量基线；老版本文档先迁移再返回。"""

        raw = self._store.read(BASELINE_DOC)
        if not isinstance(raw, dict):
            return None
        migrated = self._migrate_baseline(raw)
        value = migrated.get("baseline_kg_s")
        if not isinstance(value, (int, float)):
            return None
        return migrated

    @staticmethod
    def _migrate_baseline(raw: dict) -> dict:
        """把基线文档逐代迁移到当前版本。"""

        version = int(raw.get("version", 1))
        if version > BASELINE_VERSION:
            raise StorageError(
                f"基线文档版本 {version} 高于当前支持的 {BASELINE_VERSION}，无法迁移",
                component="store",
            )
        doc = dict(raw)
        if version < 2:
            # v1 → v2：只有基线与代际；容差置空由调用方回退出厂值，联锁默认投入。
            doc = {
                "version": 2,
                "baseline_kg_s": doc.get("baseline_kg_s"),
                "epoch": int(doc.get("epoch", 0)),
                "tolerance_kg_s": None,
                "active": True,
                "hold_reason": "",
                "ts": 0.0,
            }
            version = 2
        return doc

    def save_calibrations(self, records: list[dict]) -> str:
        """落盘标定台账（只留最近若干次）。"""

        return self._store.write(
            CALIBRATIONS_DOC,
            {"version": CALIBRATIONS_VERSION, "records": list(records)[-LEDGER_LIMIT:]},
        )

    def read_calibrations(self) -> list[dict]:
        """读回标定台账；没有台账时返回空表。"""

        raw = self._store.read(CALIBRATIONS_DOC)
        if not isinstance(raw, dict):
            return []
        version = int(raw.get("version", 1))
        if version > CALIBRATIONS_VERSION:
            raise StorageError(
                f"标定台账版本 {version} 高于当前支持的 {CALIBRATIONS_VERSION}，无法迁移",
                component="store",
            )
        records = raw.get("records")
        if not isinstance(records, list):
            return []
        return [dict(item) for item in records if isinstance(item, dict)]

    def documents(self) -> list[str]:
        """返回盐侧已落盘的文档列表。"""

        known = (LEVELS_DOC, BASELINE_DOC, CALIBRATIONS_DOC)
        return [doc for doc in self._store.list_ids() if doc in known]
