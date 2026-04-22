# -*- coding: utf-8 -*-
"""
ETF 资金流向 & 流动性监测模块 v2
==================================
改进点（v1 → v2）：
  - 移除对不稳定 net_inflow 字段的依赖
  - 改用 换手率相对强度 + 量价配合 判断资金流向
  - 新增 ETF相对强弱（vs 沪深300）识别板块轮动
  - 所有指标均来自 fund_etf_spot_em() 单次调用，境外IP友好

核心逻辑
--------
1. 换手率（Turnover Rate）
     turnover = 今日成交额(亿) / ETF规模(亿) × 100 (%)
     → 自动适配不同规模ETF，300亿和30亿ETF可以公平比较

2. 相对换手率（Relative Turnover）
     rel_turnover = 该ETF换手率 / 池内中位换手率
     → > 2.0 表示今日活跃度是平均水平的2倍，属于异常放量

3. 量价配合信号
     rel_turnover + price_chg 组合判断：
       放量上涨 → 资金主动买入（bullish）
       放量下跌 → 资金主动卖出（bearish）
       缩量    → 市场对该ETF兴趣低，流动性差

4. ETF相对强弱（vs 300）
     rs = ETF今日涨跌幅 - 沪深300今日涨跌幅
     → 正值：跑赢大盘，资金轮入；负值：跑输大盘，资金轮出

数据源：ak.fund_etf_spot_em（东方财富ETF实时行情）
"""

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 宽基ETF（国家队主要操作标的，放量信号额外标注）
_BROAD_BASE_ETFS = {"510300", "510500", "159915", "588000", "159659", "512800"}

# 相对换手率阈值
_REL_STRONG   = 2.0   # 相对换手率 ≥ 2 → 显著放量
_REL_MILD     = 1.4   # 相对换手率 ≥ 1.4 → 温和放量
_REL_QUIET    = 0.45  # 相对换手率 ≤ 0.45 → 显著缩量

# 价格变动阈值（配合量能判断方向，百分比）
_PRICE_UP     = 0.3   # 涨超 0.3% 才算上涨放量（过滤横盘噪声）
_PRICE_DOWN   = -0.3  # 跌超 0.3% 才算下跌放量

# 相对强弱阈值（vs 沪深300，百分比）
_RS_STRONG    = 0.8   # 跑赢300超过0.8% → 明显强势
_RS_WEAK      = -0.8  # 跑输300超过0.8% → 明显弱势


def _fetch_etf_spot() -> pd.DataFrame:
    """
    拉取ETF实时行情（东方财富），返回标准化 DataFrame。
    成功：含 code/name/scale/amount/chg_pct 列
    失败：返回空 DataFrame
    """
    try:
        import akshare as ak
        df = ak.fund_etf_spot_em()
        if df is None or df.empty:
            return pd.DataFrame()
        return df
    except Exception as exc:
        logger.warning("[资金流v2] ETF实时数据获取失败: %s", exc)
        return pd.DataFrame()


def _normalize_spot(spot_df: pd.DataFrame) -> pd.DataFrame:
    """
    识别并统一列名，转换数值类型。
    返回含 code/name/scale/amount/chg_pct 的 DataFrame（缺失列为 NaN）。
    """
    col_map: dict = {}
    for col in spot_df.columns:
        s = str(col)
        if "代码" in s or s.lower() == "symbol":
            col_map["code"] = col
        elif "名称" in s:
            col_map["name"] = col
        elif "规模" in s or "资产" in s:
            col_map["scale"] = col
        elif "成交额" in s:
            col_map["amount"] = col
        elif "涨跌幅" in s or "涨跌%" in s:
            col_map["chg_pct"] = col

    if "code" not in col_map:
        logger.warning("[资金流v2] 无法识别ETF代码列，跳过分析")
        return pd.DataFrame()

    df = spot_df.rename(columns={v: k for k, v in col_map.items()}).copy()
    df["code"] = df["code"].astype(str).str.strip().str.zfill(6)

    for col in ["scale", "amount", "chg_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    return df


def _get_hs300_chg(spot_df: pd.DataFrame) -> float:
    """从 spot_df 里取沪深300（510300）当日涨跌幅作为基准，找不到返回 NaN。"""
    row = spot_df[spot_df["code"] == "510300"]
    if row.empty or pd.isna(row["chg_pct"].iloc[0]):
        return float("nan")
    return float(row["chg_pct"].iloc[0])


def get_etf_fund_flow(codes: List[str]) -> pd.DataFrame:
    """
    分析 ETF 池资金流向与流动性。

    Returns
    -------
    DataFrame，列：
        code | flow_signal | label | turnover | rel_turnover | rs | is_broad_base
    按 flow_signal 强度排序（strong_in 优先）
    """
    spot_raw = _fetch_etf_spot()

    # 若获取失败，返回空结果结构
    if spot_raw.empty:
        return pd.DataFrame({
            "code":         codes,
            "flow_signal":  ["neutral"] * len(codes),
            "label":        [""] * len(codes),
            "turnover":     [np.nan] * len(codes),
            "rel_turnover": [np.nan] * len(codes),
            "rs":           [np.nan] * len(codes),
            "is_broad_base": [c in _BROAD_BASE_ETFS for c in codes],
        })

    spot = _normalize_spot(spot_raw)
    if spot.empty:
        return pd.DataFrame({
            "code":         codes,
            "flow_signal":  ["neutral"] * len(codes),
            "label":        [""] * len(codes),
            "turnover":     [np.nan] * len(codes),
            "rel_turnover": [np.nan] * len(codes),
            "rs":           [np.nan] * len(codes),
            "is_broad_base": [c in _BROAD_BASE_ETFS for c in codes],
        })

    # 只保留池内ETF
    pool = spot[spot["code"].isin(codes)].copy()

    # ── 计算换手率 ────────────────────────────────────────────
    # amount 单位：元（东财）→ 转亿；scale 单位：亿
    # 若 amount 量级过大（>1e8）认为是元，否则已经是亿
    if pool["amount"].median() > 1e8:
        pool["amount_yi"] = pool["amount"] / 1e8
    else:
        pool["amount_yi"] = pool["amount"]

    # turnover(%) = 成交额(亿) / 规模(亿) × 100
    pool["turnover"] = np.where(
        (pool["scale"] > 0) & pool["scale"].notna(),
        pool["amount_yi"] / pool["scale"] * 100,
        np.nan,
    )

    # 若规模缺失，退化为池内相对成交额排名（用成交额替代换手率）
    if pool["turnover"].isna().all():
        logger.info("[资金流v2] 规模字段缺失，改用成交额相对排名")
        pool["turnover"] = pool["amount_yi"]

    # 相对换手率 = 个股换手率 / 池内中位数换手率
    median_to = pool["turnover"].median()
    if median_to and median_to > 0:
        pool["rel_turnover"] = pool["turnover"] / median_to
    else:
        pool["rel_turnover"] = np.nan

    # ── ETF 相对强弱（vs 沪深300）────────────────────────────
    hs300_chg = _get_hs300_chg(pool)
    pool["rs"] = pool["chg_pct"] - hs300_chg  # NaN - NaN = NaN，安全

    # ── 信号判断 ──────────────────────────────────────────────
    records = []
    for code in codes:
        is_broad = code in _BROAD_BASE_ETFS
        row = pool[pool["code"] == code]

        if row.empty:
            records.append({
                "code": code, "flow_signal": "neutral", "label": "",
                "turnover": np.nan, "rel_turnover": np.nan,
                "rs": np.nan, "is_broad_base": is_broad,
            })
            continue

        r          = row.iloc[0]
        rel_to     = r["rel_turnover"] if not pd.isna(r["rel_turnover"]) else 1.0
        chg        = r["chg_pct"]      if not pd.isna(r["chg_pct"])      else 0.0
        rs_val     = r["rs"]           if not pd.isna(r["rs"])           else float("nan")
        to_val     = r["turnover"]     if not pd.isna(r["turnover"])     else float("nan")
        name       = r.get("name", code) if "name" in r.index else code

        # ── 量价配合信号 ──────────────────────────────────────
        if rel_to >= _REL_STRONG:
            if chg >= _PRICE_UP:
                signal = "strong_in"
                prefix = "🏛️国家队" if is_broad else "🏦机构"
                label  = f"{prefix}大幅放量 {rel_to:.1f}x↑{chg:+.1f}%"
            elif chg <= _PRICE_DOWN:
                signal = "strong_out"
                label  = f"⚠️大幅放量砸盘 {rel_to:.1f}x{chg:+.1f}%"
            else:
                signal = "active"     # 放量但价格横盘，多空争夺
                label  = f"🔄放量博弈 {rel_to:.1f}x（方向不明）"
        elif rel_to >= _REL_MILD:
            if chg >= _PRICE_UP:
                signal = "mild_in"
                label  = f"温和放量上涨 {rel_to:.1f}x↑{chg:+.1f}%"
            elif chg <= _PRICE_DOWN:
                signal = "mild_out"
                label  = f"温和放量下跌 {rel_to:.1f}x{chg:+.1f}%"
            else:
                signal = "neutral"
                label  = ""
        elif rel_to <= _REL_QUIET:
            signal = "quiet"
            label  = f"🔕缩量观望 {rel_to:.1f}x（流动性差）"
        else:
            signal = "neutral"
            label  = ""

        # ── 相对强弱补充标注 ──────────────────────────────────
        if not pd.isna(rs_val):
            if rs_val >= _RS_STRONG and signal not in ("strong_out", "mild_out"):
                label += f"  💪跑赢300 {rs_val:+.1f}%"
            elif rs_val <= _RS_WEAK and signal not in ("strong_in", "mild_in"):
                label += f"  🐢跑输300 {rs_val:+.1f}%"

        records.append({
            "code":         code,
            "flow_signal":  signal,
            "label":        label.strip(),
            "turnover":     round(to_val, 3) if not pd.isna(to_val) else np.nan,
            "rel_turnover": round(rel_to, 2),
            "rs":           round(rs_val, 2) if not pd.isna(rs_val) else np.nan,
            "is_broad_base": is_broad,
        })

    df = pd.DataFrame(records)

    # 排序：strong_in > mild_in > active > quiet > neutral > mild_out > strong_out
    order = {"strong_in": 0, "mild_in": 1, "active": 2,
             "quiet": 3, "neutral": 4, "mild_out": 5, "strong_out": 6}
    df["_sort"] = df["flow_signal"].map(order).fillna(4)
    df = df.sort_values("_sort").drop(columns=["_sort"]).reset_index(drop=True)
    return df


def summarize_fund_flow(flow_df: pd.DataFrame) -> str:
    """生成资金流向一句话摘要，用于报告顶部。"""
    if flow_df.empty:
        return ""
    from src.data.stock_mapping import STOCK_NAME_MAP

    def _fmt(row) -> str:
        code = row["code"]
        name = STOCK_NAME_MAP.get(code, code)
        label = row.get("label", "")
        return f"{name}({code})" + (f"[{label}]" if label else "")

    strong_in  = flow_df[flow_df["flow_signal"] == "strong_in"]
    strong_out = flow_df[flow_df["flow_signal"] == "strong_out"]
    quiet_cnt  = (flow_df["flow_signal"] == "quiet").sum()
    parts = []

    if not strong_in.empty:
        broad_in = strong_in[strong_in["is_broad_base"]]
        prefix = "🏛️国家队大幅放量" if not broad_in.empty else "🏦机构大幅放量"
        items  = "、".join(strong_in.apply(_fmt, axis=1).tolist())
        parts.append(f"{prefix}：{items}")
    if not strong_out.empty:
        items = "、".join(strong_out.apply(_fmt, axis=1).tolist())
        parts.append(f"⚠️放量砸盘：{items}")
    if quiet_cnt >= 10:
        parts.append(f"🔕{quiet_cnt}只ETF缩量观望，市场情绪低迷")

    return "　｜　".join(parts) if parts else ""
