# -*- coding: utf-8 -*-
"""
市场情绪模块
============
聚合三类量化情绪数据，为宏观评价提供有实际意义的输入：

1. 北向资金（沪深港通）
   - 当日净流入总量（亿元）
   - 5日累计净流入
   - 连续流入/流出天数

2. 涨停/跌停情绪比
   - 涨停家数 / 跌停家数
   - 比值 > 5 → 情绪过热；< 1.5 → 情绪低迷

3. 两融余额
   - 最新余额（亿元）
   - 5日变化量（增加=加杠杆，减少=去杠杆）

数据源：AKShare（全免费接口）
错误处理：任意单项失败不影响其他项，全部失败返回空dict
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any

import pandas as pd

logger = logging.getLogger(__name__)


def _get_northbound_flow() -> Dict[str, Any]:
    """
    获取北向资金（沪股通+深股通合计）净流入数据。
    返回：today_net（亿元）、5d_net、streak（连续天数，正=流入，负=流出）
    """
    try:
        import akshare as ak
        # 沪深港通资金流向（东方财富）
        df = ak.stock_em_hsgt_north_net_flow_in(indicator="北向资金")
        if df is None or df.empty:
            return {}
        df = df.sort_values(df.columns[0]).tail(10).reset_index(drop=True)
        # 列名：日期, 当日资金流入, 当日资金流入-沪股通, 当日资金流入-深股通
        net_col = [c for c in df.columns if "当日" in str(c) and "沪" not in str(c) and "深" not in str(c)]
        if not net_col:
            net_col = df.columns[1:2].tolist()
        series = pd.to_numeric(df[net_col[0]], errors="coerce") / 1e8  # 转亿元

        today_net = round(float(series.iloc[-1]), 2)
        net_5d    = round(float(series.tail(5).sum()), 2)

        # 连续流入/流出天数
        streak = 0
        direction = 1 if series.iloc[-1] >= 0 else -1
        for v in reversed(series.tolist()):
            if (v >= 0 and direction == 1) or (v < 0 and direction == -1):
                streak += direction
            else:
                break

        return {"today_net": today_net, "net_5d": net_5d, "streak": streak}
    except Exception as exc:
        logger.warning(f"[情绪] 北向资金获取失败: {exc}")
        return {}


def _get_limit_stats() -> Dict[str, Any]:
    """
    获取今日A股涨停/跌停家数及情绪比值。
    比值 = 涨停数 / max(跌停数, 1)
    """
    try:
        import akshare as ak
        today = datetime.now().strftime("%Y%m%d")
        # 涨停池
        zt = ak.stock_zt_pool_em(date=today)
        zt_count = len(zt) if zt is not None else 0
        # 跌停池
        dt = ak.stock_dt_pool_em(date=today)
        dt_count = len(dt) if dt is not None else 0

        ratio = round(zt_count / max(dt_count, 1), 1)
        if ratio >= 5:
            sentiment = "过热"
        elif ratio >= 3:
            sentiment = "偏热"
        elif ratio >= 1.5:
            sentiment = "中性"
        elif ratio >= 0.8:
            sentiment = "偏冷"
        else:
            sentiment = "恐慌"

        return {"zt": zt_count, "dt": dt_count, "ratio": ratio, "sentiment": sentiment}
    except Exception as exc:
        logger.warning(f"[情绪] 涨停跌停数据获取失败: {exc}")
        return {}


def _get_margin_balance() -> Dict[str, Any]:
    """
    获取两融余额（全市场融资余额）最新值及5日变化。
    余额增加 → 加杠杆（情绪偏多）；减少 → 去杠杆（情绪偏空）
    """
    try:
        import akshare as ak
        df = ak.stock_margin_account_info(
            start_date=(datetime.now() - timedelta(days=30)).strftime("%Y%m%d"),
            end_date=datetime.now().strftime("%Y%m%d"),
        )
        if df is None or df.empty:
            return {}
        # 找余额列（通常是"融资余额"）
        balance_col = [c for c in df.columns if "融资余额" in str(c) and "融券" not in str(c)]
        if not balance_col:
            balance_col = df.columns[1:2].tolist()
        df = df.sort_values(df.columns[0]).tail(10).reset_index(drop=True)
        series = pd.to_numeric(df[balance_col[0]], errors="coerce") / 1e8  # 亿元

        latest   = round(float(series.iloc[-1]), 0)
        change5d = round(float(series.iloc[-1] - series.iloc[-min(5, len(series))]), 0)
        trend    = "↑加杠杆" if change5d > 0 else "↓去杠杆"

        return {"balance": latest, "change5d": change5d, "trend": trend}
    except Exception as exc:
        logger.warning(f"[情绪] 两融余额获取失败: {exc}")
        return {}


def get_market_sentiment() -> Dict[str, Any]:
    """
    获取市场情绪综合数据。
    任何子项失败均不影响整体，失败项返回空dict。

    Returns
    -------
    dict with keys:
        northbound  : {today_net, net_5d, streak}
        limit       : {zt, dt, ratio, sentiment}
        margin      : {balance, change5d, trend}
        summary     : str  一句话情绪描述
        error       : str or None
    """
    result: Dict[str, Any] = {
        "northbound": {},
        "limit":      {},
        "margin":     {},
        "summary":    "",
        "error":      None,
    }

    result["northbound"] = _get_northbound_flow()
    result["limit"]      = _get_limit_stats()
    result["margin"]     = _get_margin_balance()

    # ── 生成一句话情绪总结 ─────────────────────────────────────
    parts = []

    nb = result["northbound"]
    if nb:
        flow_str  = f"北向{'+' if nb['today_net'] >= 0 else ''}{nb['today_net']}亿"
        streak    = nb.get("streak", 0)
        streak_str = f"（连续{abs(streak)}日{'流入' if streak > 0 else '流出'}）" if abs(streak) >= 2 else ""
        parts.append(flow_str + streak_str)

    lm = result["limit"]
    if lm:
        parts.append(f"涨停{lm['zt']}家/跌停{lm['dt']}家（情绪{lm['sentiment']}，比值{lm['ratio']}）")

    mg = result["margin"]
    if mg:
        parts.append(f"两融{mg['balance']:.0f}亿 {mg['trend']}")

    result["summary"] = "　｜　".join(parts) if parts else "情绪数据暂不可用"

    return result
