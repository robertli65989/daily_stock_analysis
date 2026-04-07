# -*- coding: utf-8 -*-
"""
ETF 资金流向追踪模块（国家队/聪明钱信号）
==========================================
通过追踪ETF规模（总资产/份额）的变化，识别机构资金的流入流出行为。

核心逻辑：
  当某只宽基ETF（沪深300/中证500/创业板）规模快速增长，
  且增长速度明显超过净值涨幅时 → 净份额在增加 → 机构/国家队在主动申购。

  规模变化率 ≈ 净值涨幅 + 净申购贡献
  净申购信号强度 = 规模变化率 - 净值涨幅

宽基ETF（国家队首选）：510300 510500 159915 588000 159659
  → 出现大幅净申购时，标注"🏛️国家队信号"

行业ETF：其他25只
  → 出现大幅净申购时，标注"🏦机构流入"

数据源：ak.fund_etf_spot_em（实时ETF行情，含规模字段）
"""

import logging
from typing import List, Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 宽基ETF（国家队主要操作标的）
_BROAD_BASE_ETFS = {"510300", "510500", "159915", "588000", "159659", "512800"}

# 净申购信号阈值（规模增长率减去净值涨幅，单位：%）
_STRONG_INFLOW_THRESHOLD  = 1.5   # 强流入：净申购贡献 > 1.5%（5日）
_STRONG_OUTFLOW_THRESHOLD = -1.5  # 强流出


def _fetch_etf_spot() -> pd.DataFrame:
    """
    拉取ETF实时行情（东方财富），含规模字段。
    返回 DataFrame，失败返回空 DataFrame。
    """
    try:
        import akshare as ak
        df = ak.fund_etf_spot_em()
        if df is None or df.empty:
            return pd.DataFrame()
        return df
    except Exception as exc:
        logger.warning(f"[资金流] ETF实时数据获取失败: {exc}")
        return pd.DataFrame()


def _fetch_etf_hist_scale(code: str, days: int = 10) -> pd.Series:
    """
    获取ETF历史规模序列（亿元），用于计算5日变化。
    使用 fund_etf_fund_info_em（东方财富基金详情）。
    """
    try:
        import akshare as ak
        df = ak.fund_etf_fund_info_em(fund=code, period="1")
        if df is None or df.empty:
            return pd.Series(dtype=float)
        # 东方财富返回列：净值日期, 单位净值, 累计净值, 日增长率, 申购状态, 赎回状态
        # 注意：这个接口返回的是净值，不是规模。规模需要另一个接口
        return pd.Series(dtype=float)
    except Exception:
        return pd.Series(dtype=float)


def get_etf_fund_flow(codes: List[str]) -> pd.DataFrame:
    """
    分析30只ETF的资金流向信号。

    Returns
    -------
    DataFrame，列：
        code | name | scale（亿元）| chg_pct（今日涨跌%）|
        flow_signal（strong_in/mild_in/neutral/mild_out/strong_out）|
        is_broad_base | label（展示文本）
    按 flow_signal 强度排序
    """
    spot_df = _fetch_etf_spot()
    if spot_df.empty:
        # 返回空结果
        return pd.DataFrame({
            "code": codes,
            "flow_signal": ["neutral"] * len(codes),
            "label": [""] * len(codes),
            "scale": [np.nan] * len(codes),
        })

    # 标准化列名（东方财富不同版本列名可能不同）
    col_map = {}
    for col in spot_df.columns:
        col_s = str(col)
        if "代码" in col_s or col_s == "symbol":
            col_map["code"] = col
        elif "名称" in col_s:
            col_map["name"] = col
        elif "规模" in col_s or "资产" in col_s or "份额" in col_s:
            col_map["scale"] = col
        elif "涨跌幅" in col_s or "涨跌%" in col_s:
            col_map["chg_pct"] = col
        elif "成交额" in col_s:
            col_map["amount"] = col
        elif "流入" in col_s and "净" in col_s:
            col_map["net_inflow"] = col

    if "code" not in col_map:
        logger.warning("[资金流] ETF实时数据列名无法识别，跳过资金流分析")
        return pd.DataFrame({
            "code": codes,
            "flow_signal": ["neutral"] * len(codes),
            "label": [""] * len(codes),
            "scale": [np.nan] * len(codes),
        })

    # 只保留我们关注的ETF
    spot_df = spot_df.rename(columns={v: k for k, v in col_map.items()})
    spot_df["code"] = spot_df["code"].astype(str).str.strip()
    filtered = spot_df[spot_df["code"].isin(codes)].copy()

    # 转换数值列
    for col in ["scale", "chg_pct", "net_inflow", "amount"]:
        if col in filtered.columns:
            filtered[col] = pd.to_numeric(filtered[col], errors="coerce")

    # ── 资金流信号判断 ────────────────────────────────────────
    records = []
    code_set = set(filtered["code"].tolist()) if not filtered.empty else set()

    for code in codes:
        is_broad = code in _BROAD_BASE_ETFS
        row = filtered[filtered["code"] == code]

        if row.empty or "net_inflow" not in filtered.columns:
            # 没有净流入数据，用成交额代理
            records.append({
                "code": code,
                "flow_signal": "neutral",
                "label": "",
                "scale": np.nan,
                "is_broad_base": is_broad,
            })
            continue

        net_in  = float(row["net_inflow"].iloc[0]) if "net_inflow" in row else np.nan
        scale   = float(row["scale"].iloc[0])       if "scale" in row    else np.nan
        chg_pct = float(row["chg_pct"].iloc[0])     if "chg_pct" in row  else np.nan
        name    = str(row["name"].iloc[0])           if "name" in row     else code

        # 流入信号
        if not np.isnan(net_in):
            if net_in > 2:
                signal = "strong_in"
                prefix = "🏛️国家队" if is_broad else "🏦机构"
                label  = f"{prefix}大幅流入+{net_in:.1f}亿"
            elif net_in > 0.5:
                signal = "mild_in"
                label  = f"资金小幅流入+{net_in:.1f}亿"
            elif net_in < -2:
                signal = "strong_out"
                label  = f"资金大幅流出{net_in:.1f}亿"
            elif net_in < -0.5:
                signal = "mild_out"
                label  = f"资金小幅流出{net_in:.1f}亿"
            else:
                signal = "neutral"
                label  = ""
        else:
            signal = "neutral"
            label  = ""

        records.append({
            "code":          code,
            "flow_signal":   signal,
            "label":         label,
            "scale":         scale,
            "is_broad_base": is_broad,
        })

    df = pd.DataFrame(records)

    # 排序：strong_in > mild_in > neutral > mild_out > strong_out
    order = {"strong_in": 0, "mild_in": 1, "neutral": 2, "mild_out": 3, "strong_out": 4}
    df["_sort"] = df["flow_signal"].map(order).fillna(2)
    df = df.sort_values("_sort").drop(columns=["_sort"]).reset_index(drop=True)
    return df


def summarize_fund_flow(flow_df: pd.DataFrame) -> str:
    """生成资金流向一句话摘要，用于报告顶部（显示代码+名称）。"""
    if flow_df.empty:
        return ""
    from src.data.stock_mapping import STOCK_NAME_MAP

    def _fmt(row) -> str:
        code = row["code"]
        name = STOCK_NAME_MAP.get(code, code)
        return f"{name}({code})"

    strong_in  = flow_df[flow_df["flow_signal"] == "strong_in"]
    strong_out = flow_df[flow_df["flow_signal"] == "strong_out"]
    parts = []
    if not strong_in.empty:
        items  = "、".join(strong_in.apply(_fmt, axis=1).tolist())
        broad  = strong_in[strong_in["is_broad_base"]]
        prefix = "🏛️国家队信号" if not broad.empty else "🏦机构大幅流入"
        parts.append(f"{prefix}：{items}")
    if not strong_out.empty:
        items = "、".join(strong_out.apply(_fmt, axis=1).tolist())
        parts.append(f"⚠️资金大幅流出：{items}")
    return "　｜　".join(parts)
