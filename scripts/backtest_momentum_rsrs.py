# -*- coding: utf-8 -*-
"""
回测：动量轮动 + RSRS 择时策略
================================
策略逻辑：
  - 每周五收盘计算信号，下周一开盘执行
  - RSRS > 0：买入动量排名前3的ETF，等权分配
  - RSRS ≤ 0：清仓，全仓 511090（短债，避险）
  - 任一持仓跌破买入价 -5%：当日止损，转为短债

数据范围：2020-01-01 至 2025-12-31
初始资金：20,000 元

运行方式：
  cd daily_stock_analysis
  python scripts/backtest_momentum_rsrs.py

输出：
  - 逐年收益 vs 沪深300基准
  - 总收益、年化收益、最大回撤、夏普比率
  - 持仓变化日志（可选，设 VERBOSE=True）
"""

import sys
import time
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────────────────────
START_DATE      = "20200101"
END_DATE        = "20251231"
INIT_CAPITAL    = 20_000.0
TOP_N           = 3          # 动量前N只
LOOKBACK_SHORT  = 20         # 短期动量窗口（交易日）
LOOKBACK_LONG   = 60         # 长期动量窗口（交易日）
RSRS_N          = 18         # RSRS OLS 窗口
RSRS_M          = 600        # RSRS Z-score 历史窗口
STOP_LOSS_PCT   = 0.05       # 硬止损 5%
SAFE_HAVEN      = "511090"   # 避险品种（短债）
BENCHMARK       = "510300"   # 基准（沪深300 ETF）
CSI300_CODE     = "000300"   # 沪深300指数（用于RSRS）
VERBOSE         = False      # True：打印每次调仓日志

ETF_POOL = [
    "510300","510500","159915","588000","510880","159659",
    "512800","512000","512690","159928","512010","159883",
    "512480","515390","515880","515790","516160","512660",
    "516970","518880","515220","561150","513050","513100",
    "513500","511090","159845","512980","516220","516950",
]

# ──────────────────────────────────────────────────────────────
# 数据获取
# ──────────────────────────────────────────────────────────────

def _sina_prefix(code: str) -> str:
    """ETF 代码转新浪前缀格式：5/6开头=sh，1/0/3开头=sz"""
    return ("sh" if code[0] in ("5", "6") else "sz") + code


def fetch_etf_close(code: str, start: str, end: str) -> pd.Series:
    """获取 ETF 前复权收盘价（日线）。东方财富失败自动切新浪。"""
    import akshare as ak

    # 主数据源：东方财富（前复权）
    try:
        df = ak.stock_zh_a_hist(
            symbol=code, period="daily",
            start_date=start, end_date=end, adjust="qfq",
        )
        if df is not None and not df.empty:
            df["日期"] = pd.to_datetime(df["日期"])
            return df.set_index("日期")["收盘"].rename(code)
    except Exception:
        pass

    # 备用：新浪 ETF 历史（不复权，但用于回测相对比较可接受）
    try:
        sina_sym = _sina_prefix(code)
        df2 = ak.fund_etf_hist_sina(symbol=sina_sym)
        if df2 is not None and not df2.empty:
            df2["date"] = pd.to_datetime(df2["date"])
            s = df2.set_index("date")["close"].rename(code)
            s = s[s.index >= pd.to_datetime(start)]
            return s
    except Exception as e2:
        print(f"  ⚠ {code} 新浪备用也失败: {e2}", file=sys.stderr)

    return pd.Series(dtype=float, name=code)


def fetch_index_close(start: str, end: str) -> pd.Series:
    """获取沪深300指数收盘价（用于RSRS）。"""
    try:
        import akshare as ak
        # 主力：index_zh_a_hist
        df = ak.index_zh_a_hist(symbol="000300", period="daily",
                                 start_date=start, end_date=end)
        if df is not None and not df.empty:
            df["日期"] = pd.to_datetime(df["日期"])
            return df.set_index("日期")["收盘"].rename("csi300")
    except Exception:
        pass
    try:
        # 备用：stock_zh_index_daily
        import akshare as ak
        df = ak.stock_zh_index_daily(symbol="sh000300")
        if df is not None and not df.empty:
            df.index = pd.to_datetime(df.index)
            s = df["close"].rename("csi300")
            return s[s.index >= start]
    except Exception as e:
        print(f"  ⚠ 沪深300指数数据获取失败: {e}", file=sys.stderr)
    return pd.Series(dtype=float, name="csi300")


def fetch_index_ohlc(start: str, end: str) -> pd.DataFrame:
    """获取沪深300 high/low（用于RSRS OLS）。与 market_timing.py 逻辑一致。"""
    import akshare as ak
    import time as _time

    df = None
    for attempt in range(3):
        try:
            raw = ak.index_zh_a_hist(symbol="000300", period="daily",
                                      start_date=start, end_date=end)
            raw = raw.rename(columns={"日期": "date", "最高": "high", "最低": "low",
                                       "开盘": "open", "收盘": "close"})
            raw["date"] = pd.to_datetime(raw["date"])
            df = raw.set_index("date").sort_index()[["high", "low", "open", "close"]]
            break
        except Exception as exc:
            print(f"  index_zh_a_hist 第{attempt+1}次失败: {exc}", file=sys.stderr)
            if attempt < 2:
                _time.sleep(3)

    if df is None:
        try:
            raw2 = ak.stock_zh_index_daily(symbol="sh000300")
            raw2["date"] = pd.to_datetime(raw2["date"])
            raw2 = raw2.set_index("date").sort_index()
            df = raw2[["open", "high", "low", "close"]]
            print("  [备用] 使用新浪数据源")
        except Exception as exc2:
            print(f"  [ERROR] 沪深300 OHLC 全部失败: {exc2}", file=sys.stderr)
            return pd.DataFrame()

    return df


# ──────────────────────────────────────────────────────────────
# RSRS 计算（钝化版简化：只用 β zscore，不加 R² 钝化以减复杂度）
# ──────────────────────────────────────────────────────────────

def calc_rsrs_series(ohlc: pd.DataFrame, n: int = RSRS_N, m: int = RSRS_M) -> pd.Series:
    """
    计算全历史 RSRS β zscore 序列。
    ohlc: DataFrame with columns [high, low], date index, sorted ascending.
    返回 pd.Series，index 为日期，值为 zscore。
    """
    ohlc = ohlc.dropna()
    if len(ohlc) < n + m:
        return pd.Series(dtype=float)

    betas = []
    dates = []
    for i in range(n - 1, len(ohlc)):
        window = ohlc.iloc[i - n + 1: i + 1]
        x = window["low"].values
        y = window["high"].values
        # OLS: y = β·x + α
        xm, ym = x.mean(), y.mean()
        denom = ((x - xm) ** 2).sum()
        if denom == 0:
            betas.append(np.nan)
        else:
            beta = ((x - xm) * (y - ym)).sum() / denom
            betas.append(beta)
        dates.append(ohlc.index[i])

    beta_series = pd.Series(betas, index=dates)

    # 滚动 Z-score
    zscore = (beta_series - beta_series.rolling(m).mean()) / beta_series.rolling(m).std()
    return zscore.dropna()


# ──────────────────────────────────────────────────────────────
# 动量计算
# ──────────────────────────────────────────────────────────────

def calc_momentum_scores(close_df: pd.DataFrame, as_of: pd.Timestamp,
                          short: int = LOOKBACK_SHORT, long: int = LOOKBACK_LONG) -> pd.Series:
    """
    截至 as_of 日期，计算各 ETF 的综合动量评分（0-100）。
    close_df: 宽表，index=日期，columns=ETF代码
    """
    sub = close_df[close_df.index <= as_of]

    ret_short = {}
    ret_long  = {}
    for code in close_df.columns:
        s = sub[code].dropna()
        if len(s) >= short + 1:
            ret_short[code] = s.iloc[-1] / s.iloc[-short] - 1
        if len(s) >= long + 1:
            ret_long[code]  = s.iloc[-1] / s.iloc[-long]  - 1

    df = pd.DataFrame({"r20": ret_short, "r60": ret_long})
    valid = df.dropna()
    if len(valid) < 2:
        return pd.Series(dtype=float)

    n = len(valid)
    valid["p20"] = valid["r20"].rank(ascending=True) / n
    valid["p60"] = valid["r60"].rank(ascending=True) / n
    valid["score"] = (valid["p20"] * 0.5 + valid["p60"] * 0.5) * 100
    return valid["score"].sort_values(ascending=False)


# ──────────────────────────────────────────────────────────────
# 回测引擎
# ──────────────────────────────────────────────────────────────

def run_backtest(
    close_df: pd.DataFrame,
    rsrs_series: pd.Series,
    init_capital: float = INIT_CAPITAL,
) -> dict:
    """
    主回测循环。

    Returns dict with keys:
      nav_series, trades, log
    """
    all_dates = close_df.index.sort_values()
    nav       = init_capital
    positions = {}   # {code: {"shares": float, "cost": float}}
    nav_history = []
    trades = []
    log = []

    def _nav_today(date):
        """当日净值 = 现金 + 市值"""
        total = cash
        for c, pos in positions.items():
            if c in close_df.columns and date in close_df.index:
                px = close_df.at[date, c]
                if not np.isnan(px):
                    total += pos["shares"] * px
        return total

    cash = init_capital

    # 找出每周五（周5=4）的日期
    fridays = [d for d in all_dates if d.weekday() == 4]

    # 当前目标持仓（信号，周一生效）
    target_portfolio = {SAFE_HAVEN: 1.0}   # 初始空仓持短债

    for i, date in enumerate(all_dates):
        # ── 0. 止损检查（每日开盘前）─────────────────────────
        stop_triggered = []
        for code, pos in list(positions.items()):
            if code == SAFE_HAVEN:
                continue
            if date not in close_df.index:
                continue
            px = close_df.at[date, code]
            if np.isnan(px):
                continue
            if px <= pos["cost"] * (1 - STOP_LOSS_PCT):
                # 止损卖出
                proceeds = pos["shares"] * px
                cash += proceeds
                stop_triggered.append(code)
                trades.append({"date": date, "action": "STOP", "code": code,
                                "price": px, "shares": pos["shares"]})
                if VERBOSE:
                    log.append(f"{date.date()} 止损 {code} @{px:.4f}（成本{pos['cost']:.4f}）")
        for code in stop_triggered:
            del positions[code]

        # 如果止损后仓位变化，补仓短债
        if stop_triggered:
            # 把现金移到短债
            if SAFE_HAVEN in close_df.columns and date in close_df.index:
                sh_px = close_df.at[date, SAFE_HAVEN]
                if not np.isnan(sh_px) and sh_px > 0:
                    shares = cash / sh_px
                    if SAFE_HAVEN in positions:
                        positions[SAFE_HAVEN]["shares"] += shares
                    else:
                        positions[SAFE_HAVEN] = {"shares": shares, "cost": sh_px}
                    cash = 0.0

        # ── 1. 周五生成新信号 ──────────────────────────────────
        if date in fridays:
            rsrs_val = rsrs_series.get(date, np.nan)

            # 找下一个交易日（周一或节后首日）执行
            future_dates = all_dates[all_dates > date]
            exec_date = future_dates[0] if len(future_dates) > 0 else None

            if np.isnan(rsrs_val):
                # 无 RSRS 信号，维持短债
                new_target = {SAFE_HAVEN: 1.0}
            elif rsrs_val > 0:
                # 多头：动量前 TOP_N
                scores = calc_momentum_scores(close_df, date)
                # 排除短债本身
                scores = scores[scores.index != SAFE_HAVEN]
                top = scores.head(TOP_N).index.tolist()
                if top:
                    new_target = {c: 1.0 / len(top) for c in top}
                else:
                    new_target = {SAFE_HAVEN: 1.0}
            else:
                # 空头：全仓短债
                new_target = {SAFE_HAVEN: 1.0}

            # 调仓将在下一交易日执行
            if exec_date is not None:
                target_portfolio = {"_exec_date": exec_date, **new_target}

            if VERBOSE and not np.isnan(rsrs_val):
                desc = "多头" if rsrs_val > 0 else "空头"
                log.append(f"{date.date()} 信号({desc}, RSRS={rsrs_val:.3f}) → {list(new_target.keys())}")

        # ── 2. 执行调仓（信号对应执行日）─────────────────────
        exec_date = target_portfolio.get("_exec_date")
        if exec_date is not None and date == exec_date:
            new_weights = {k: v for k, v in target_portfolio.items() if k != "_exec_date"}

            # 计算总资产
            total_assets = cash
            for code, pos in positions.items():
                if date in close_df.index and code in close_df.columns:
                    px = close_df.at[date, code]
                    if not np.isnan(px):
                        total_assets += pos["shares"] * px

            # 卖出不在新目标中的持仓
            for code in list(positions.keys()):
                if code not in new_weights:
                    if date in close_df.index and code in close_df.columns:
                        px = close_df.at[date, code]
                        if not np.isnan(px):
                            cash += positions[code]["shares"] * px
                            trades.append({"date": date, "action": "SELL", "code": code,
                                           "price": px, "shares": positions[code]["shares"]})
                    del positions[code]

            # 买入/调整目标持仓
            for code, weight in new_weights.items():
                target_val = total_assets * weight
                if date in close_df.index and code in close_df.columns:
                    px = close_df.at[date, code]
                    if np.isnan(px) or px <= 0:
                        continue
                    cur_val = positions[code]["shares"] * px if code in positions else 0
                    diff_val = target_val - cur_val
                    diff_shares = diff_val / px
                    if abs(diff_shares) < 0.01:
                        continue
                    cash -= diff_shares * px
                    if code in positions:
                        new_shares = positions[code]["shares"] + diff_shares
                        if new_shares > 0:
                            # 更新成本（加权平均）
                            old_cost = positions[code]["cost"]
                            old_sh   = positions[code]["shares"]
                            if diff_shares > 0:
                                new_cost = (old_cost * old_sh + px * diff_shares) / new_shares
                            else:
                                new_cost = old_cost
                            positions[code] = {"shares": new_shares, "cost": new_cost}
                        else:
                            del positions[code]
                    else:
                        if diff_shares > 0:
                            positions[code] = {"shares": diff_shares, "cost": px}
                    action = "BUY" if diff_shares > 0 else "TRIM"
                    trades.append({"date": date, "action": action, "code": code,
                                   "price": px, "shares": abs(diff_shares)})

            target_portfolio = {k: v for k, v in target_portfolio.items() if k != "_exec_date"}

            if VERBOSE:
                total_now = _nav_today(date)
                log.append(f"{date.date()} 调仓完成，持仓: {list(positions.keys())}，总资产: {total_now:.0f}")

        # ── 3. 记录当日净值 ────────────────────────────────────
        nav_history.append({"date": date, "nav": _nav_today(date)})

    nav_series = pd.DataFrame(nav_history).set_index("date")["nav"]
    return {"nav_series": nav_series, "trades": pd.DataFrame(trades), "log": log}


# ──────────────────────────────────────────────────────────────
# 绩效统计
# ──────────────────────────────────────────────────────────────

def calc_metrics(nav: pd.Series, benchmark: pd.Series, rf: float = 0.02) -> dict:
    """计算回测绩效指标。"""
    daily_ret  = nav.pct_change().dropna()
    bench_ret  = benchmark.pct_change().dropna()

    # 对齐
    common = daily_ret.index.intersection(bench_ret.index)
    dr = daily_ret.loc[common]
    br = bench_ret.loc[common]

    total_days  = len(dr)
    years       = total_days / 252
    total_ret   = nav.iloc[-1] / nav.iloc[0] - 1
    ann_ret     = (1 + total_ret) ** (1 / years) - 1

    # 最大回撤
    cum_max     = nav.cummax()
    drawdowns   = nav / cum_max - 1
    max_dd      = drawdowns.min()

    # 夏普
    excess      = dr - rf / 252
    sharpe      = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0

    # 基准
    bench_total = benchmark.iloc[-1] / benchmark.iloc[0] - 1
    bench_ann   = (1 + bench_total) ** (1 / years) - 1

    # 胜率（日）
    win_rate = (dr > br).mean()

    # 逐年收益
    yearly = {}
    for year in sorted(nav.index.year.unique()):
        year_nav = nav[nav.index.year == year]
        if len(year_nav) < 2:
            continue
        yr = year_nav.iloc[-1] / year_nav.iloc[0] - 1
        year_bm = benchmark[benchmark.index.year == year]
        if len(year_bm) >= 2:
            br_yr = year_bm.iloc[-1] / year_bm.iloc[0] - 1
        else:
            br_yr = np.nan
        yearly[year] = {"strategy": yr, "benchmark": br_yr, "alpha": yr - br_yr}

    return {
        "total_ret":   total_ret,
        "ann_ret":     ann_ret,
        "max_dd":      max_dd,
        "sharpe":      sharpe,
        "win_rate":    win_rate,
        "bench_total": bench_total,
        "bench_ann":   bench_ann,
        "yearly":      yearly,
    }


def print_report(metrics: dict, trades_df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("  动量轮动 + RSRS 择时 · 回测报告")
    print("=" * 60)
    print(f"  时间范围：{START_DATE} → {END_DATE}")
    print(f"  初始资金：¥{INIT_CAPITAL:,.0f}")
    print(f"  ETF池：{len(ETF_POOL)} 只，TOP {TOP_N} 轮动，避险 {SAFE_HAVEN}")
    print(f"  止损线：{STOP_LOSS_PCT*100:.0f}%，RSRS 窗口 N={RSRS_N} M={RSRS_M}")
    print()
    print("  【总体绩效】")
    print(f"  策略总收益  ：{metrics['total_ret']*100:+.1f}%")
    print(f"  策略年化收益：{metrics['ann_ret']*100:+.1f}%")
    print(f"  基准总收益  ：{metrics['bench_total']*100:+.1f}%（沪深300 ETF）")
    print(f"  基准年化收益：{metrics['bench_ann']*100:+.1f}%")
    print(f"  最大回撤    ：{metrics['max_dd']*100:.1f}%")
    print(f"  夏普比率    ：{metrics['sharpe']:.2f}")
    print(f"  日胜率(vs基准)：{metrics['win_rate']*100:.1f}%")
    print()
    print("  【逐年收益】")
    print(f"  {'年份':>6}  {'策略':>8}  {'基准':>8}  {'超额':>8}")
    print(f"  {'----':>6}  {'------':>8}  {'------':>8}  {'------':>8}")
    for yr, v in metrics["yearly"].items():
        bm_str  = f"{v['benchmark']*100:+.1f}%" if not np.isnan(v['benchmark']) else "  N/A "
        alp_str = f"{v['alpha']*100:+.1f}%"     if not np.isnan(v['benchmark']) else "  N/A "
        print(f"  {yr:>6}  {v['strategy']*100:>+7.1f}%  {bm_str:>8}  {alp_str:>8}")
    print()

    if not trades_df.empty:
        n_trades  = len(trades_df)
        n_stop    = (trades_df["action"] == "STOP").sum() if "action" in trades_df else 0
        print(f"  【交易统计】调仓 {n_trades} 次，其中止损 {n_stop} 次")
        # 最常持有的ETF
        if "code" in trades_df:
            top_held = trades_df[trades_df["action"].isin(["BUY","TRIM"])]["code"].value_counts().head(5)
            if not top_held.empty:
                print(f"  最常买入 TOP5：{', '.join(top_held.index.tolist())}")
    print("=" * 60)


# ──────────────────────────────────────────────────────────────
# 主程序
# ──────────────────────────────────────────────────────────────

def main():
    print("[*] 正在获取历史数据（约需 2-5 分钟）...")

    # 预留足够的历史数据用于 RSRS（M=600 需要约 2.4 年）
    data_start = "20170101"

    # 1. 获取 OHLC（RSRS）
    print("  → 沪深300 OHLC（RSRS计算）...")
    ohlc = fetch_index_ohlc(data_start, END_DATE)
    if ohlc.empty:
        print("  ✗ 沪深300 OHLC 数据获取失败，无法计算 RSRS，程序终止")
        sys.exit(1)

    # 2. 获取 ETF 收盘价
    print(f"  → {len(ETF_POOL)} 只 ETF 收盘价...")
    close_dict = {}
    for i, code in enumerate(ETF_POOL):
        sys.stdout.write(f"\r     {i+1}/{len(ETF_POOL)} {code}    ")
        sys.stdout.flush()
        s = fetch_etf_close(code, data_start, END_DATE)
        if not s.empty:
            close_dict[code] = s
        time.sleep(0.3)   # 控速避免限流
    print()

    if not close_dict:
        print("  ✗ ETF 数据全部获取失败，程序终止")
        sys.exit(1)

    available = list(close_dict.keys())
    print(f"  ✓ 成功获取 {len(available)}/{len(ETF_POOL)} 只 ETF")

    close_df = pd.DataFrame(close_dict)
    close_df.index = pd.to_datetime(close_df.index)
    close_df = close_df.sort_index()

    # 截取回测范围
    close_df = close_df[close_df.index >= START_DATE]

    # 3. 基准（510300）
    bm_series = close_df[BENCHMARK] if BENCHMARK in close_df.columns else pd.Series(dtype=float)

    # 4. 计算 RSRS
    print("  → 计算 RSRS 序列...")
    ohlc.index = pd.to_datetime(ohlc.index)
    rsrs_series = calc_rsrs_series(ohlc, n=RSRS_N, m=RSRS_M)
    print(f"  ✓ RSRS 序列长度 {len(rsrs_series)}，最新值 {rsrs_series.iloc[-1]:.4f}")

    # 5. 运行回测
    print("\n[*] 开始回测...")
    result = run_backtest(close_df, rsrs_series)
    nav = result["nav_series"]

    if VERBOSE and result["log"]:
        print("\n【调仓日志】")
        for line in result["log"][-50:]:   # 只打印最后50条
            print(" ", line)

    # 6. 绩效统计
    if bm_series.empty:
        print("  ⚠ 找不到 510300 数据，基准统计将不准确")
        bm_series = pd.Series(1.0, index=nav.index)

    # 对齐基准
    bm_aligned = bm_series.reindex(nav.index).ffill().bfill()
    metrics = calc_metrics(nav, bm_aligned)

    print_report(metrics, result["trades"])

    # 7. 保存净值曲线
    out_path = "scripts/backtest_nav.csv"
    nav_df = pd.DataFrame({"nav": nav, "benchmark": bm_aligned})
    nav_df.to_csv(out_path)
    print(f"\n[OK] 净值曲线已保存至 {out_path}")
    print("   可用 Excel 或 matplotlib 绘图可视化")


if __name__ == "__main__":
    main()
