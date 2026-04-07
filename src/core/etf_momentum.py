# -*- coding: utf-8 -*-
"""
ETF 动量因子模块
================
计算30只ETF的短/长期动量排名，用于辅助轮动选股。

核心逻辑：
- 短期动量（ret20）：近20个交易日收益率
- 长期动量（ret60）：近60个交易日收益率
- 相对强度（rel_str）：ETF收益率 - 沪深300收益率（超额）
- 综合动量评分（0-100）：50% × 短期百分位 + 50% × 长期百分位
- 排名（rank）：1 = 最强动量

数据源：ak.stock_zh_a_hist（前复权），与主分析模块相同接口
并发获取（ThreadPoolExecutor），控制速率避免触发限流
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_LOOKBACK_SHORT   = 20     # 短期动量窗口（交易日）
_LOOKBACK_LONG    = 60     # 长期动量窗口（交易日）
_CALENDAR_BUFFER  = 110    # 自然日：60交易日约需90自然日，加余量
_MAX_WORKERS      = 5      # 并发获取线程数
_RETRY_SLEEP      = 2      # 重试间隔秒数
_BENCHMARK        = "000300"


def _fetch_close(code: str, calendar_days: int = _CALENDAR_BUFFER) -> Optional[pd.Series]:
    """
    获取单只ETF前复权收盘价序列。
    使用 stock_zh_a_hist（A股/ETF通用接口）。
    失败返回 None，不抛出异常。
    """
    try:
        import akshare as ak
        end_date   = datetime.now()
        start_date = end_date - timedelta(days=calendar_days)
        df = ak.stock_zh_a_hist(
            symbol     = code,
            period     = "daily",
            start_date = start_date.strftime("%Y%m%d"),
            end_date   = end_date.strftime("%Y%m%d"),
            adjust     = "qfq",
        )
        if df is None or df.empty:
            return None
        df = df.rename(columns={"日期": "date", "收盘": "close"})
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date").sort_index()["close"]
    except Exception as exc:
        logger.debug(f"[动量] ETF {code} 数据获取失败: {exc}")
        return None


def _fetch_benchmark_close(calendar_days: int = _CALENDAR_BUFFER) -> Optional[pd.Series]:
    """获取沪深300收盘价（复用 market_timing 的双数据源逻辑）。"""
    try:
        from src.core.market_timing import _fetch_index_ohlc
        df = _fetch_index_ohlc(symbol=_BENCHMARK, lookback_days=calendar_days)
        return df["close"]
    except Exception as exc:
        logger.warning(f"[动量] 基准数据获取失败: {exc}")
        return None


def _calc_return(series: pd.Series, n: int) -> float:
    """计算最近 n 个交易日的收益率（%）。数据不足返回 nan。"""
    if series is None or len(series) < n + 1:
        return np.nan
    return round((series.iloc[-1] / series.iloc[-n] - 1) * 100, 2)


def get_etf_momentum(
    codes: List[str],
    lookback_short: int = _LOOKBACK_SHORT,
    lookback_long:  int = _LOOKBACK_LONG,
) -> pd.DataFrame:
    """
    批量计算ETF动量排名。

    Parameters
    ----------
    codes : ETF代码列表（6位数字）
    lookback_short : 短期动量窗口（交易日）
    lookback_long  : 长期动量窗口（交易日）

    Returns
    -------
    DataFrame，列：code | ret20 | ret60 | rel_str | momentum_score | rank
    按 momentum_score 降序排列（rank=1 为最强）
    """
    calendar_days = max(lookback_long, lookback_short) * 2 + 30  # 保守估计

    # ── 并发拉取所有ETF数据 ────────────────────────────────────
    close_map: dict[str, Optional[pd.Series]] = {}
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
        future_map = {pool.submit(_fetch_close, code, calendar_days): code for code in codes}
        for future in as_completed(future_map):
            code = future_map[future]
            try:
                close_map[code] = future.result()
            except Exception as exc:
                logger.debug(f"[动量] {code} 线程异常: {exc}")
                close_map[code] = None

    # ── 基准收益率 ─────────────────────────────────────────────
    bm_close  = _fetch_benchmark_close(calendar_days)
    bm_ret20  = _calc_return(bm_close, lookback_short) if bm_close is not None else np.nan
    bm_ret60  = _calc_return(bm_close, lookback_long)  if bm_close is not None else np.nan

    # ── 逐只计算指标 ───────────────────────────────────────────
    records = []
    for code in codes:
        series  = close_map.get(code)
        ret20   = _calc_return(series, lookback_short)
        ret60   = _calc_return(series, lookback_long)
        rel_str = round(ret20 - bm_ret20, 2) if not (np.isnan(ret20) or np.isnan(bm_ret20)) else np.nan
        records.append({
            "code":    code,
            "ret20":   ret20,
            "ret60":   ret60,
            "rel_str": rel_str,
        })

    df = pd.DataFrame(records)

    # ── 动量评分：百分位排名加权 ───────────────────────────────
    n = len(df)
    # rank() 越大 = 收益越高；转为 0-1 百分位
    df["pct20"] = df["ret20"].rank(ascending=True, na_option="bottom") / n
    df["pct60"] = df["ret60"].rank(ascending=True, na_option="bottom") / n
    df["momentum_score"] = ((df["pct20"] * 0.5 + df["pct60"] * 0.5) * 100).round(1)

    # 综合排名（1=最强）
    df["rank"] = df["momentum_score"].rank(ascending=False, method="min", na_option="bottom").astype(int)

    df = df.sort_values("momentum_score", ascending=False).reset_index(drop=True)
    return df[["code", "ret20", "ret60", "rel_str", "momentum_score", "rank"]]
