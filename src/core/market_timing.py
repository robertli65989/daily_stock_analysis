# -*- coding: utf-8 -*-
"""
大盘择时模块 v2
===============
五指标投票机制，平衡滞后性与抗噪性：

  指标              维度            有效滞后
  ──────────────── ─────────────── ────────
  RSRS 钝化版       结构性支撑强度   ~12天
  鳄鱼线            慢趋势方向       ~21天
  MA20/MA60位置     中期趋势方向     ~5天   ← 新增
  ROC20 动量        价格变化速率     ~2天   ← 新增
  RSI(14)           超买/超卖       ~3天   ← 新增

投票规则（每指标 +1/0/-1）：
  总分 ≥ 3  → 满仓
  总分 1~2  → 半仓
  总分 ≤ 0  → 空仓

空仓 override（风控红线，优先于投票）：
  RSRS 钝化值 < -0.7  OR  鳄鱼吃饱  → 强制空仓

数据来源：AKShare index_zh_a_hist（东财）→ stock_zh_index_daily（新浪备用）
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─── 常量 ───────────────────────────────────────────────────────────────────
_RSRS_N      = 18     # OLS 滚动窗口（交易日）
_RSRS_M      = 600    # Z-score 历史窗口（交易日，约2.4年）
_RSRS_UPPER  = 0.7    # 空仓 override 上阈值（投票用 ±0.3）
_RSRS_LOWER  = -0.7   # 空仓 override 下阈值

_MA_SHORT    = 20
_MA_LONG     = 60

_ROC_PERIOD  = 20     # 20 交易日动量
_ROC_BULL    = 3.0    # ROC > 3% → 多头票
_ROC_BEAR    = -3.0   # ROC < -3% → 空头票

_RSI_PERIOD  = 14
_RSI_BULL    = 60     # RSI > 60 → 多头票
_RSI_BEAR    = 40     # RSI < 40 → 空头票

_ALLIGATOR_JAW_PERIOD   = 13
_ALLIGATOR_JAW_SHIFT    = 8
_ALLIGATOR_TEETH_PERIOD = 8
_ALLIGATOR_TEETH_SHIFT  = 5
_ALLIGATOR_LIPS_PERIOD  = 5
_ALLIGATOR_LIPS_SHIFT   = 3

_SYMBOL = "000300"  # 沪深300 作为大盘代理


# ─── 数据获取 ────────────────────────────────────────────────────────────────

def _fetch_index_ohlc(symbol: str = _SYMBOL, lookback_days: int = 1200) -> pd.DataFrame:
    """
    通过 AKShare 获取指数日线 OHLCV，返回 DataFrame（date 为索引）。
    lookback_days 至少 RSRS_M + RSRS_N = 618，设 1200（约840交易日）以留足余量。
    """
    try:
        import akshare as ak
    except ImportError:
        raise ImportError("akshare 未安装，请运行: pip install akshare")

    import time as _time
    end_date   = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)

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
            df  = raw[["open", "high", "low", "close", "volume"]]
            break
        except Exception as exc:
            last_exc = exc
            logger.warning("index_zh_a_hist 第%d次失败: %s", attempt + 1, exc)
            if attempt < 2:
                _time.sleep(3)

    # ── 备用数据源：stock_zh_index_daily（新浪，不过滤日期取全量）──
    if df is None:
        logger.warning("主数据源全部失败，切换新浪备用: %s", last_exc)
        try:
            sina_symbol = f"sh{symbol}" if not symbol.startswith(("sh", "sz")) else symbol
            raw2 = ak.stock_zh_index_daily(symbol=sina_symbol)
            raw2["date"] = pd.to_datetime(raw2["date"])
            raw2 = raw2.set_index("date").sort_index()
            df = raw2[["open", "high", "low", "close", "volume"]]
        except Exception as exc2:
            logger.error("备用数据源也失败: %s", exc2)
            raise last_exc

    return df


# ─── RSRS ───────────────────────────────────────────────────────────────────

def _calc_rsrs(df: pd.DataFrame, n: int = _RSRS_N, m: int = _RSRS_M) -> pd.DataFrame:
    """RSRS 钝化版（passive）。结果列：beta, r2, zscore, passive"""
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant

    high = df["high"].values
    low  = df["low"].values
    size = len(df)

    betas, r2s = np.full(size, np.nan), np.full(size, np.nan)
    for i in range(n - 1, size):
        y = high[i - n + 1: i + 1]
        x = add_constant(low[i - n + 1: i + 1])
        res   = OLS(y, x).fit()
        betas[i] = res.params[1]
        r2s[i]   = res.rsquared

    df = df.copy()
    df["beta"] = betas
    df["r2"]   = r2s

    beta_s = df["beta"]
    df["zscore"] = (beta_s - beta_s.rolling(m).mean()) / beta_s.rolling(m).std()

    ret      = df["close"].pct_change()
    ret_std  = ret.rolling(n).std()
    ret_rank = ret_std.rolling(m).rank(pct=True).fillna(0.5)
    df["passive"] = df["zscore"] * (df["r2"] ** (4 * ret_rank))

    return df


def _rsrs_vote(rsrs_val: float) -> int:
    """RSRS 投票：用 ±0.3 作为软阈值（override 仍用 ±0.7）"""
    if pd.isna(rsrs_val):
        return 0
    if rsrs_val > 0.3:
        return 1
    if rsrs_val < -0.3:
        return -1
    return 0


# ─── 鳄鱼线 ─────────────────────────────────────────────────────────────────

def _smma(series: pd.Series, period: int) -> pd.Series:
    """平滑移动均线（Wilder's MA），alpha = 1/period。"""
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


def _calc_alligator(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["close"]
    df["jaw"]   = _smma(close, _ALLIGATOR_JAW_PERIOD).shift(_ALLIGATOR_JAW_SHIFT)
    df["teeth"] = _smma(close, _ALLIGATOR_TEETH_PERIOD).shift(_ALLIGATOR_TEETH_SHIFT)
    df["lips"]  = _smma(close, _ALLIGATOR_LIPS_PERIOD).shift(_ALLIGATOR_LIPS_SHIFT)
    return df


def _alligator_state(lips: float, teeth: float, jaw: float) -> str:
    if lips > teeth > jaw:
        return "bull"
    if lips < teeth < jaw:
        return "bear"
    return "neutral"


def _alligator_vote(state: str) -> int:
    return {"bull": 1, "bear": -1, "neutral": 0}.get(state, 0)


# ─── MA20/MA60 趋势 ──────────────────────────────────────────────────────────

def _calc_ma_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算 MA20、MA60 并判断趋势。
    结果列：ma20, ma60, ma_state（'bull'/'bear'/'neutral'）
    """
    df = df.copy()
    df["ma20"] = df["close"].rolling(_MA_SHORT).mean()
    df["ma60"] = df["close"].rolling(_MA_LONG).mean()

    def _state(row) -> str:
        c, m20, m60 = row["close"], row["ma20"], row["ma60"]
        if pd.isna(m20) or pd.isna(m60):
            return "neutral"
        if c > m20 > m60:
            return "bull"
        if c < m20 < m60:
            return "bear"
        return "neutral"

    df["ma_state"] = df.apply(_state, axis=1)
    return df


def _ma_vote(state: str) -> int:
    return {"bull": 1, "bear": -1, "neutral": 0}.get(state, 0)


# ─── ROC20 动量 ──────────────────────────────────────────────────────────────

def _calc_roc(df: pd.DataFrame, period: int = _ROC_PERIOD) -> pd.DataFrame:
    """20 日价格变化率（%）。"""
    df = df.copy()
    df["roc"] = df["close"].pct_change(period) * 100
    return df


def _roc_vote(roc: float) -> int:
    if pd.isna(roc):
        return 0
    if roc > _ROC_BULL:
        return 1
    if roc < _ROC_BEAR:
        return -1
    return 0


# ─── RSI(14) ─────────────────────────────────────────────────────────────────

def _calc_rsi(df: pd.DataFrame, period: int = _RSI_PERIOD) -> pd.DataFrame:
    """Wilder RSI。"""
    df   = df.copy()
    diff = df["close"].diff()
    gain = diff.clip(lower=0)
    loss = (-diff).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - 100 / (1 + rs)
    return df


def _rsi_vote(rsi: float) -> int:
    if pd.isna(rsi):
        return 0
    if rsi > _RSI_BULL:
        return 1
    if rsi < _RSI_BEAR:
        return -1
    return 0


# ─── 对外接口 ─────────────────────────────────────────────────────────────────

def get_market_timing(symbol: str = _SYMBOL) -> Dict[str, Any]:
    """
    五指标投票择时，返回结构化结果。

    Returns dict:
      rsrs_signal       : float  RSRS 钝化值
      alligator_state   : str    'bull'/'bear'/'neutral'
      ma_state          : str    'bull'/'bear'/'neutral'
      roc               : float  20日动量(%)
      rsi               : float  RSI(14)
      vote_detail       : dict   各指标得票明细
      vote_total        : int    总分
      final_position    : int    0=空仓 1=半仓 2=满仓
      override_reason   : str or None  强制空仓原因
      jaw/teeth/lips    : float  鳄鱼三线
      ma20/ma60         : float  均线值
      summary           : str    一句话结论
      error             : str or None
    """
    result: Dict[str, Any] = {
        "rsrs_signal":      None,
        "alligator_state":  "neutral",
        "ma_state":         "neutral",
        "roc":              None,
        "rsi":              None,
        "vote_detail":      {},
        "vote_total":       0,
        "final_position":   1,   # 数据不足时默认不干预
        "override_reason":  None,
        "jaw": None, "teeth": None, "lips": None,
        "ma20": None, "ma60": None,
        "summary": "大盘择时数据不足，不干预ETF仓位",
        "error":  None,
    }

    try:
        df = _fetch_index_ohlc(symbol)
        if len(df) < _RSRS_M + _RSRS_N:
            result["error"] = "历史数据不足"
            return result

        # ── 计算五个指标 ──────────────────────────────────────
        df = _calc_rsrs(df)
        df = _calc_alligator(df)
        df = _calc_ma_trend(df)
        df = _calc_roc(df)
        df = _calc_rsi(df)

        last = df.iloc[-1]

        rsrs_val  = float(last["passive"]) if not pd.isna(last["passive"]) else float("nan")
        jaw_v     = float(last["jaw"])
        teeth_v   = float(last["teeth"])
        lips_v    = float(last["lips"])
        ali_state = _alligator_state(lips_v, teeth_v, jaw_v)
        ma_state  = str(last["ma_state"])
        ma20_v    = float(last["ma20"]) if not pd.isna(last["ma20"]) else float("nan")
        ma60_v    = float(last["ma60"]) if not pd.isna(last["ma60"]) else float("nan")
        roc_v     = float(last["roc"])  if not pd.isna(last["roc"])  else float("nan")
        rsi_v     = float(last["rsi"])  if not pd.isna(last["rsi"])  else float("nan")

        # ── 各指标投票 ────────────────────────────────────────
        v_rsrs = _rsrs_vote(rsrs_val)
        v_ali  = _alligator_vote(ali_state)
        v_ma   = _ma_vote(ma_state)
        v_roc  = _roc_vote(roc_v)
        v_rsi  = _rsi_vote(rsi_v)
        total  = v_rsrs + v_ali + v_ma + v_roc + v_rsi

        vote_detail = {
            "RSRS":     v_rsrs,
            "鳄鱼线":   v_ali,
            "MA趋势":   v_ma,
            "ROC20":    v_roc,
            "RSI14":    v_rsi,
        }

        # ── 风控红线（override 优先于投票）────────────────────
        override_reason = None
        if not pd.isna(rsrs_val) and rsrs_val < _RSRS_LOWER:
            override_reason = f"RSRS={rsrs_val:.3f} < {_RSRS_LOWER}，结构性空头"
        elif ali_state == "bear":
            override_reason = "鳄鱼吃饱（三线空头排列），强制空仓"

        # ── 综合仓位 ──────────────────────────────────────────
        if override_reason:
            final = 0
            summary = f"🔴 大盘空仓（风控触发）：{override_reason}"
        elif total >= 3:
            final = 2
            summary = f"🟢 大盘满仓（投票 {total}/5）：多指标共振看多，ETF轮动正常运行"
        elif total >= 1:
            final = 1
            summary = f"🟡 大盘半仓（投票 {total}/5）：信号分歧，谨慎介入"
        else:
            final = 0
            summary = f"🔴 大盘空仓（投票 {total}/5）：多空力量偏弱，建议观望"

        result.update({
            "rsrs_signal":      round(rsrs_val, 4) if not pd.isna(rsrs_val) else None,
            "alligator_state":  ali_state,
            "ma_state":         ma_state,
            "roc":              round(roc_v, 2)  if not pd.isna(roc_v)  else None,
            "rsi":              round(rsi_v, 1)  if not pd.isna(rsi_v)  else None,
            "vote_detail":      vote_detail,
            "vote_total":       total,
            "final_position":   final,
            "override_reason":  override_reason,
            "jaw":              round(jaw_v,   2),
            "teeth":            round(teeth_v, 2),
            "lips":             round(lips_v,  2),
            "ma20":             round(ma20_v, 2) if not pd.isna(ma20_v) else None,
            "ma60":             round(ma60_v, 2) if not pd.isna(ma60_v) else None,
            "summary":          summary,
        })

    except Exception as e:
        logger.warning("大盘择时计算失败（不影响ETF分析）: %s", e)
        result["error"]   = str(e)
        result["summary"] = f"大盘择时计算异常（{e}），不干预ETF仓位"

    return result
