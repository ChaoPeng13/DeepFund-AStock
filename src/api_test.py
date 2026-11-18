"""AKShare A股数据接口冒烟验证脚本."""

import time
from datetime import datetime
from apis.router import Router, APISource

# 测试用 ticker: em_fast / cninfo 任一有近期增减持即可命中
TICKER = "688012"
TRADING_DATE = datetime.strptime("2026-11-16", "%Y-%m-%d")

# 同时验证一个老票（Tier 1 + Tier 2 都空时走 Tier 3 em_full 回退）
FALLBACK_TICKER = "600519"


def check(name, func):
    """Run one smoke check and print its result."""
    print(f"\n===== {name} =====")
    t0 = time.time()
    try:
        result = func()
        elapsed = time.time() - t0
        print(f"[OK] ({elapsed:.2f}s)")
        if isinstance(result, list):
            print(f"     items={len(result)}")
            if result:
                print(f"     sample={result[0]}")
        elif isinstance(result, str):
            print(result[:300])
        else:
            print(result)
        return True, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        print(f"[FAIL] ({elapsed:.2f}s) {type(e).__name__}: {e}")
        return False, elapsed


# =============================================================================
# 开始
# =============================================================================
print("=" * 60)
print("AKShare A股接口冒烟验证")
print("=" * 60)

router = Router(APISource.AKSHARE)

results = {}

results["daily_candles_df"] = check(
    "daily candles df",
    lambda: router.get_a_stock_daily_candles_df(ticker=TICKER, trading_date=TRADING_DATE).tail(),
)

results["last_close_price"] = check(
    "last close price",
    lambda: router.get_a_stock_last_close_price(ticker=TICKER, trading_date=TRADING_DATE),
)

results["fundamentals"] = check(
    "fundamentals",
    lambda: router.get_a_stock_fundamentals(ticker=TICKER, trading_date=TRADING_DATE).model_dump_json(indent=2),
)

results["company_news"] = check(
    "company news",
    lambda: [m.model_dump_json() for m in router.get_a_stock_news(
        ticker=TICKER, trading_date=TRADING_DATE, news_count=5)],
)

results["policy_news"] = check(
    "policy news",
    lambda: [m.model_dump_json() for m in router.get_a_stock_policy_news(
        trading_date=TRADING_DATE, news_count=5)],
)

# insider trades: em_fast (180 天窗口, ~6s) → cninfo (~5s) → em_full (~210s)
results[f"insider_trades[{TICKER}]"] = check(
    f"insider trades [{TICKER}]",
    lambda: [t.model_dump_json() for t in router.get_a_stock_insider_trades(
        ticker=TICKER, trading_date=TRADING_DATE, limit=5)],
)

results[f"insider_trades[{FALLBACK_TICKER}] (em_full fallback)"] = check(
    f"insider trades [{FALLBACK_TICKER}] (em_full fallback)",
    lambda: [t.model_dump_json() for t in router.get_a_stock_insider_trades(
        ticker=FALLBACK_TICKER, trading_date=datetime(2025, 4, 1), limit=5)],
)

results["economic_indicators"] = check(
    "economic indicators",
    lambda: router.get_a_stock_economic_indicators().model_dump_json(indent=2),
)

# =============================================================================
# 汇总
# =============================================================================
print("\n" + "=" * 60)
print("汇总")
print("=" * 60)

total_elapsed = 0

for name, (ok, elapsed) in results.items():
    tag = "OK" if ok else "FAILED"
    print(f"  [{tag}] {elapsed:7.2f}s  {name}")
    total_elapsed += elapsed

print(f"\n总耗时: {total_elapsed:.2f}s")
print("=" * 60)
