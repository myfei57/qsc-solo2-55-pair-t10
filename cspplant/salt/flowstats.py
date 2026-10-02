"""实测流量波动统计。

容差不能拍脑袋：工况平稳时波动小，容差应收窄以减少误报；工况摇摆时波动大，
容差应放宽以避免漏报。这里用 EWMA 同时跟流量均值与方差，只在"带负荷在运"
时喂样本（停机与切换的流量不算正常运行波动），样本不足时不报数，由调用方
回退到种子容差。

口径固定：波动用总体标准差（EWMA 方差开方），容差 = k*sigma 后夹到配置的
上下限。有效期与作废判据都不在这里——这里只回答"最近的运行波动有多大"。
"""

from __future__ import annotations

from ..config import SaltLimits

# 喂够这么多样本后，波动估计才被采信；不足时调用方用种子容差。
MIN_SAMPLES = 30


class FlowStats:
    """按 EWMA 跟踪在运流量的均值与波动。"""

    def __init__(self, limits: SaltLimits) -> None:
        self._limits = limits
        self._mean: float | None = None
        self._var: float = 0.0
        self._samples = 0
        self._resets = 0

    @property
    def limits(self) -> SaltLimits:
        return self._limits

    def observe(self, flow_kg_s: float) -> None:
        """喂一个在运流量样本，更新均值与方差。"""

        value = float(flow_kg_s)
        alpha = self._limits.tolerance_ewma_alpha
        if self._mean is None:
            self._mean = value
            self._var = 0.0
        else:
            delta = value - self._mean
            self._mean += alpha * delta
            # EWMA 方差：（1-alpha）加权旧方差 + alpha 加权当前离差平方。
            self._var = (1.0 - alpha) * self._var + alpha * delta * delta
        self._samples += 1

    def sigma_kg_s(self) -> float | None:
        """返回当前波动标准差；样本不足时返回 ``None``。"""

        if self._mean is None or self._samples < MIN_SAMPLES:
            return None
        return self._var ** 0.5

    def mean_kg_s(self) -> float | None:
        """返回当前 EWMA 均值；尚无样本时返回 ``None``。"""

        return self._mean

    def tolerance_kg_s(self) -> float:
        """返回自适应容差；波动样本不足时回退到配置种子容差。"""

        sigma = self.sigma_kg_s()
        if sigma is None:
            return self._limits.flow_tolerance_kg_s
        return self._limits.adaptive_tolerance_kg_s(sigma)

    def reset(self) -> None:
        """清空统计（标定换代后重新积累）。"""

        self._mean = None
        self._var = 0.0
        self._samples = 0
        self._resets += 1

    def samples(self) -> int:
        return self._samples

    def resets(self) -> int:
        return self._resets
