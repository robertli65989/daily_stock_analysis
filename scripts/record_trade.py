"""
Plan B：手动记录交易，更新 portfolio.json。
由 .github/workflows/record_trade.yml 调用。

环境变量：
  TRADE_ACTION  buy | sell | confirm
  TRADE_CODE    ETF代码（如 515880）
  TRADE_SHARES  份数（buy 时必填）
  TRADE_PRICE   成本价（buy 时必填）
  TRADE_DATE    交易日期（可选，默认今天）
"""
import json
import os
import sys
from datetime import date
from pathlib import Path

PORTFOLIO_PATH = Path(__file__).parent.parent / "portfolio.json"


def load() -> dict:
    try:
        with open(PORTFOLIO_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"total_capital": 20000, "positions": {}}


def save(portfolio: dict) -> None:
    with open(PORTFOLIO_PATH, "w", encoding="utf-8") as f:
        json.dump(portfolio, f, ensure_ascii=False, indent=2)


def main():
    action = os.environ.get("TRADE_ACTION", "").strip().lower()
    code   = os.environ.get("TRADE_CODE",   "").strip()
    shares_str = os.environ.get("TRADE_SHARES", "").strip()
    price_str  = os.environ.get("TRADE_PRICE",  "").strip()
    trade_date = os.environ.get("TRADE_DATE",   "").strip() or date.today().strftime("%Y-%m-%d")

    if not action or not code:
        print("❌ TRADE_ACTION 和 TRADE_CODE 不能为空", file=sys.stderr)
        sys.exit(1)

    portfolio = load()
    positions = portfolio.setdefault("positions", {})

    if action == "sell":
        if code in positions:
            name = positions[code].get("name", code)
            del positions[code]
            print(f"✅ 已卖出 {name}（{code}），从持仓中移除")
        else:
            print(f"⚠️  {code} 不在当前持仓中，无需操作")

    elif action == "buy":
        if not shares_str or not price_str:
            print("❌ buy 操作需要提供 TRADE_SHARES 和 TRADE_PRICE", file=sys.stderr)
            sys.exit(1)
        try:
            shares = int(shares_str)
            price  = float(price_str)
        except ValueError:
            print("❌ 份数或价格格式错误", file=sys.stderr)
            sys.exit(1)

        existing = positions.get(code, {})
        positions[code] = {
            "name":          existing.get("name", code),
            "shares":        shares,
            "cost_price":    round(price, 4),
            "buy_date":      trade_date,
            "stop_loss_pct": existing.get("stop_loss_pct", 0.05),
            "confirmed":     True,   # 手动确认，Plan A 不覆盖
        }
        print(f"✅ 买入 {code}：{shares}份 @ {price}，已标记 confirmed=true")

    elif action == "confirm":
        if code not in positions:
            print(f"⚠️  {code} 不在当前持仓中，请先通过 buy 操作添加", file=sys.stderr)
            sys.exit(1)
        positions[code]["confirmed"] = True
        # 如果提供了实际价格/份数，同步更新
        if shares_str:
            try:
                positions[code]["shares"] = int(shares_str)
            except ValueError:
                pass
        if price_str:
            try:
                positions[code]["cost_price"] = round(float(price_str), 4)
            except ValueError:
                pass
        print(f"✅ {code} 已确认（confirmed=true），Plan A 不再自动覆盖")

    else:
        print(f"❌ 未知操作类型：{action}，支持 buy / sell / confirm", file=sys.stderr)
        sys.exit(1)

    save(portfolio)
    print(f"📋 当前持仓：{list(positions.keys())}")


if __name__ == "__main__":
    main()
