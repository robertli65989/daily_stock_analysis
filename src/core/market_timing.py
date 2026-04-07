# -*- coding: utf-8 -*-
"""
大盘择时模块
============
使用两个独立信号判断当前大盘是否适合持仓：

1. RSRS 钝化版（阻力支撑相对强度）
   - 对沪深300日线 high/low 做滚动OLS，β斜率越大说明上涨动能越强
   - 钝化版：在震荡市（收益率波动率高分位）时自动压缩信号，减少误判
   - 开仓：signal > 0.7；平仓：signal < -0.7

2. 鳄鱼线（Bill Williams Alligator）
   - 下颚（蓝）: SMMA(13) 右移8期
   - 牙齿（红）: SMMA(8)  右移5期
   - 上唇（绿）: SMMA(5)  右移3期
   - 饥饿（多头）：上唇 > 牙齿 > 下颚 → 允许开仓
   - 吃饱（空头）：上唇 < 牙齿 < 下颚 → 强制空仓
   - 沉睡（震荡）：三线纠缠 → 维持现状

综合逻辑（AND 开仓，OR 空仓）：
  - RSRS < -0.7  OR  鳄鱼吃饱       → 强制空仓
  - RSRS > 0.7   AND 鳄鱼饥饿/沉睡  → 满仓
  - RSRS > 0.7   AND 鳄鱼沉睡       → 半仓
  - 其他                             → 维持（报告中显示"谨慎"）

数据来源：AKShare index_zh_a_hist，完全免费
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─── 常量 ───────────────────────────────────────────────────
_RSRS_N = 18      # OLS 滚动窗口（交易日数）
_RSRS_M = 600     # Z-score 历史窗口（交易日数，约2.4年）
_RSRS_UPPER = 0.7
_RSRS_LOWER = -0.7

_ALLIGATOR_JAW_PERIOD   = 13
_ALLIGATOR_JAW_SHIFT    = 8
_ALLIGATOR_TEETH_PERIOD = 8
_ALLIGATOR_TEETH_SHIFT  = 5
_ALLIGATOR_LIPS_PERIOD  = 5
_ALLIGATOR_LIPS_SHIFT   = 3

_SYMBOL = "000300"   # 沪深300 作为大盘代理


# ─── 数据获取 ────────────────────────────────────────────────

def _fetch_index_ohlc(symbol: str = _SYMBOL, lookback_days: int = 900) -> pd.DataFrame:
    """
    通过 AKShare 获取指数日线 OHLCV，返回 DataFrame（date 为索引）。
    lookback_days 至少需要 RSRS_M + RSRS_N = 618，建议 900（留余量）。
    """
    try:
        import akshare as ak
    except ImportError:
        raise ImportError("akshare 未安装，请运行: pip install akshare")

    end_date   = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)

    import time as _time

    # ── 主数据源：index_zh_a_hist（东方财富，3次重试）──
    df = None
    last_exc = None
    for attempt in range(3):
        try:
            raw = ak.index_zh_a_hist(
                symbol     = symbol,
                period     = "daily",
                start_date = start_date.strftime("%Y%m%d"),
                end_date   = end_date.strftime("%Y%m%d"),
            )
            rename_map = {"日期": "date", "开盘": "open", "收盘": "close",
                          "最高": "high", "最低": "low", "成交量": "volume"}
            raw = raw.rename(columns=rename_map)
            raw["date"] = pd.to_datetime(raw["date"])
            raw = raw.set_index("date").sort_index()
            df = raw[["open", "high", "low", "close", "volume"]]
            break
        except Exception as exc:
            last_exc = exc
            logger.warning(f"index_zh_a_hist 第{attempt+1}次失败: {exc}")
            if attempt < 2:
                _time.sleep(3)

    # ── 备用数据源：stock_zh_index_daily（新浪，列名不同）──
    if df is None:
        logger.warning(f"主数据源全部失败，切换备用数据源 stock_zh_index_daily: {last_exc}")
        try:
            # 新浪接口：symbol 需加交易所前缀，如 sh000300
            sina_symbol = f"sh{symbol}" if not symbol.startswith(("sh", "sz")) else symbol
            raw2 = ak.stock_zh_index_daily(symbol=sina_symbol)
            # 新浪返回列：date, open, high, low, close, volume
            raw2["date"] = pd.to_datetime(raw2["date"])
            raw2 = raw2.set_index("date").sort_index()
            # 不过滤日期，保留全量历史，确保 RSRS M=600 滚动窗口有足够数据
            df = raw2[["open", "high", "low", "close", "volume"]]
        except Exception as exc2:
            logger.error(f"备用数据源也失败: {exc2}")
            raise last_exc  # 抛出原始错误以保留上下文

    return df


# ─── RSRS 计算 ───────────────────────────────────────────────

def _calc_rsrs(df: pd.DataFrame, n: int = _RSRS_N, m: int = _RSRS_M) -> pd.DataFrame:
    """
    计算 RSRS 钝化版（passive）信号。
    结果列：beta, r2, zscore, passive
    """
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant

    high = df["high"].values
    low  = df["low"].values
    size = len(df)

    betas, r2s = np.full(size, np.nan), np.full(size, np.nan)
    for i in range(n - 1, size):
        y = high[i - n + 1: i + 1]
        x = add_constant(low[i - n + 1: i + 1])
        res = OLS(y, x).fit()
        betas[i] = res.params[1]
        r2s[i]   = res.rsquared

    df = df.copy()
    df["beta"] = betas
    df["r2"]   = r2s

    # Z-score（标准分）
    beta_s = df["beta"]
    df["zscore"] = (beta_s - beta_s.rolling(m).mean()) / beta_s.rolling(m).std()

    # 钝化权重：收益率波动率的历史分位数
    ret      = df["close"].pct_change()
    ret_std  = ret.rolling(n).std()
    ret_rank = ret_std.rolling(m).rank(pct=True).fillna(0.5)

    # 钝化版：R² 的指数随震荡程度增大，压缩信号
    df["passive"] = df["zscore"] * (df["r2"] ** (4 * ret_rank))

    return df


# ─── 鳄鱼线计算 ──────────────────────────────────────────────

def _smma(series: pd.Series, period: int) -> pd.Series:
    """平滑移动均线（SMMA / Wilder's MA），等价于 EWM alpha=1/period。"""
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


def _calc_alligator(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算鳄鱼线三条线（已包含位移）。
    列：jaw, teeth, lips（均已 shift，代表今日可见值）
    """
    close = df["close"]
    df = df.copy()
    # 各线 SMMA 计算后向右位移
    df["jaw"]   = _smma(close, _ALLIGATOR_JAW_PERIOD).shift(_ALLIGATOR_JAW_SHIFT)
    df["teeth"] = _smma(close, _ALLIGATOR_TEETH_PERIOD).shift(_ALLIGATOR_TEETH_SHIFT)
    df["lips"]  = _smma(close, _ALLIGATOR_LIPS_PERIOD).shift(_ALLIGATOR_LIPS_SHIFT)
    return df


def _alligator_state(lips: float, teeth: float, jaw: float) -> str:
    """
    返回 'bull'（饥饿）/ 'bear'（吃饱）/ 'neutral'（沉睡/纠缠）。
    """
    if lips > teeth > jaw:
        return "bull"
    if lips < teeth < jaw:
        return "bear"
    return "neutral"


# ─── 对外接口 ────────────────────────────────────────────────

def get_market_timing(symbol: str = _SYMBOL) -> Dict[str, Any]:
    """
    计算今日大盘择时信号，返回结构化结果。

    Returns:
        dict with keys:
          rsrs_signal    : float，RSRS 钝化值（最新一日）
          rsrs_position  : int，1=RSRS允许开仓, 0=RSRS不允许
          alligator_state: str，'bull'/'bear'/'neutral'
          alligator_position: int，1=鳄鱼允许, 0=鳄鱼空头禁止
          final_position : int，0=空仓, 1=半仓, 2=满仓
          jaw/teeth/lips : float，鳄鱼三线最新值
          summary        : str，一句话结论
          error          : str or None
    """
    result: Dict[str, Any] = {
        "rsrs_signal": None,
        "rsrs_position": -1,
        "alligator_state": "neutral",
        "alligator_position": 1,
        "final_position": 1,   # 数据不足时默认不干预
        "jaw": None, "teeth": None, "lips": None,
        "summary": "大盘择时数据不足，不干预ETF仓位",
        "error": None,
    }

    try:
        df = _fetch_index_ohlc(symbol)
        if len(df) < _RSRS_N + 50:
            result["error"] = "历史数据不足"
            return result

        # ── RSRS ──
        df = _calc_rsrs(df)
        rsrs_val = df["passive"].iloc[-1]

        if pd.isna(rsrs_val):
            rsrs_pos = -1          # 数据不足，不干预
        elif rsrs_val > _RSRS_UPPER:
            rsrs_pos = 2           # 强烈看多
        elif rsrs_val > 0:
            rsrs_pos = 1           # 温和看多
        elif rsrs_val > _RSRS_LOWER:
            rsrs_pos = 0           # 中性偏弱
        else:
            rsrs_pos = -2          # 看空，空仓

        result["rsrs_signal"]   = round(float(rsrs_val), 4)
        result["rsrs_position"] = rsrs_pos

        # ── 鳄鱼线 ──
        df = _calc_alligator(df)
        jaw_v    = df["jaw"].iloc[-1]
        teeth_v  = df["teeth"].iloc[-1]
        lips_v   = df["lips"].iloc[-1]
        state    = _alligator_state(lips_v, teeth_v, jaw_v)
        ali_pos  = 0 if state == "bear" else 1

        result["alligator_state"]    = state
        result["alligator_position"] = ali_pos
        result["jaw"]   = round(float(jaw_v),   2)
        result["teeth"] = round(float(teeth_v), 2)
        result["lips"]  = round(float(lips_v),  2)

        # ── 综合判断 ──
        # 空仓条件（OR）：鳄鱼吃饱 或 RSRS看空
        if state == "bear" or rsrs_pos == -2:
            final = 0
            summary = "🔴 大盘空仓信号：鳄鱼吃饱或RSRS看空，ETF全部观望"
        # 满仓条件（AND）：RSRS强烈看多 且 鳄鱼饥饿
        elif rsrs_pos == 2 and state == "bull":
            final = 2
            summary = "🟢 大盘满仓信号：RSRS强势+鳄鱼饥饿，ETF轮动正常运行"
        # 半仓条件：RSRS积极但鳄鱼不确定，或鳄鱼多头但RSRS一般
        elif rsrs_pos >= 1 or state == "bull":
            final = 1
            summary = "🟡 大盘半仓信号：信号不一致，建议半仓谨慎介入"
        else:
            final = 0
            summary = "🔴 大盘空仓信号：RSRS中性偏弱，建议观望"

        result["final_position"] = final
        result["summary"]        = summary

    except Exception as e:
        logger.warning(f"大盘择时计算失败（不影响ETF分析）: {e}")
        result["error"]   = str(e)
        result["summary"] = f"大盘择时计算异常（{e}），不干预ETF仓位"

    return result
