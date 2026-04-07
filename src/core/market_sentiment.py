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
    北向资金（沪深港通）净流入数据。
    注意：中国监管自2024年8月起停止公开披露实时北向资金数据，
    目前无可用的公开接口，返回空dict。
    保留此函数以便未来数据恢复时复用。
    """
    return {}


def _get_limit_stats() -> Dict[str, Any]:
    """
    获取今日A股涨停/跌停家数及情绪比值。
    数据源：ak.stock_market_activity_legu（乐咕，实时，含涨跌停）
    比值 = 涨停数 / max(跌停数, 1)
    """
    try:
        import akshare as ak
        df = ak.stock_market_activity_legu()
        if df is None or df.empty:
            return {}

        def _get_val(item_name):
            rows = df[df["item"] == item_name]
            if rows.empty:
                return 0
            return int(pd.to_numeric(rows["value"].iloc[0], errors="coerce") or 0)

        zt_count = _get_val("涨停")
        dt_count = _get_val("跌停")

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
    获取两融余额（沪市融资融券余额）最新值及5日变化。
    余额增加 → 加杠杆（情绪偏多）；减少 → 去杠杆（情绪偏空）
    数据源：ak.macro_china_market_margin_sh（上交所公布）
    """
    try:
        import akshare as ak
        df = ak.macro_china_market_margin_sh()
        if df is None or df.empty:
            return {}
        # 列：日期, 融资买入额, 融资余额, 融券卖出量, 融券余量, 融券余额, 融资融券余额
        df["日期"] = pd.to_datetime(df["日期"])
        df = df.sort_values("日期").tail(10).reset_index(drop=True)

        # 用"融资融券余额"（元），转亿元
        bal_col = "融资融券余额" if "融资融券余额" in df.columns else df.columns[-1]
        series  = pd.to_numeric(df[bal_col], errors="coerce") / 1e8

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
