"""
Supabase 实验数据统计分析模块
============================
数据源：真实运行 `src/main.py` 后写入 Supabase 的实验记录
（config / portfolio / decision / signal 四张表，见 database/supabase_setup.sql）。

与 `statistics.py`（基于 src/experiment/output/*.csv 离线仿真输出）不同，本模块：
  1. 直接连接 Supabase，拉取 2026-11-10 ~ 2026-11-16 的真实运行记录
  2. 处理同一交易日的重跑冗余行（每交易日仅保留 updated_at 最新的一条，
     即最终完成的那次运行，避免半途中断的组合污染序列）
  3. 复用 statistics.py 的统计原语，计算论文 Appendix D 全部指标
     （点估计 + Bootstrap 95% CI + 假设检验 p 值 + 效应量）
  4. 内置三层数据校验（论文可复现性要求）：
       A. 资产恒等式：total_assets = cashflow + Σ position.value
       B. 持仓累计一致性：第 d 日持仓 = 前 d 日决策的累积净额
       C. 决策完整性：每个交易日 ticker 数 == 配置中的 ticker 数
  5. 将原始实验记录与统计结果本地落盘为 CSV（用户要求：结果保存到本地 csv）

使用方法（工作目录 src/experiment）：
    python supabase_stats.py
    python supabase_stats.py --exp-name minimax-test --start 2026-11-10 --end 2026-11-16
"""

import argparse
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# 复用 statistics.py 的统计原语（Bootstrap CI / CAPM / 效应量 / 稳健性）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 复用项目根 src 下的数据库与配置模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib.util

# 注意：不能用 `import statistics`——本目录下的 statistics.py 会与标准库同名。
# Python 3.14 起标准库 statistics 被冻结（frozen），在 sys.meta_path 中先于
# PathFinder 生效，因此即使本目录在 sys.path[0] 也只会拿到标准库版本，
# 表现为 AttributeError: module 'statistics' has no attribute 'compute_cr_with_ci'。
# 这里按文件路径显式加载本地模块，注册为唯一名字，彻底规避同名冲突。
_STATS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "statistics.py")
_spec = importlib.util.spec_from_file_location("deepfund_experiment_stats", _STATS_PATH)
stats_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = stats_mod
_spec.loader.exec_module(stats_mod)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_supabase")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 论文 Table 6 / A股适配参数
RISK_FREE_RATE = 0.015          # 无风险利率（1年期国债）
INITIAL_CASH = 100_000.0        # 与 minimax.yaml cashflow 一致
BENCHMARK_INDEX = "000300"      # 沪深300


# ===========================================================================
# 1. Supabase 连接与数据拉取
# ===========================================================================
def get_client():
    """建立 Supabase 客户端（读取项目根目录 .env 中的 SUPABASE_URL / SUPABASE_KEY）。"""
    from dotenv import load_dotenv
    from supabase import create_client

    env_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env"
    )
    load_dotenv(env_path)
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("缺少 SUPABASE_URL / SUPABASE_KEY，请检查项目根目录 .env")
    return create_client(url, key)


def resolve_config_id(client, exp_name: str) -> str:
    """按实验名解析 config_id。"""
    resp = client.table("config").select("id,exp_name,tickers,llm_model,llm_provider") \
        .eq("exp_name", exp_name).execute()
    if not resp.data:
        raise RuntimeError(f"Supabase 中未找到实验配置: {exp_name}")
    row = resp.data[0]
    print(f"  实验配置: {row['exp_name']} | 模型: {row['llm_provider']}/{row['llm_model']}")
    print(f"  Tickers: {row['tickers']}")
    return row["id"]


def fetch_portfolio_chain(client, config_id: str, start: str, end: str) -> pd.DataFrame:
    """拉取交易日窗口内的组合序列，并消除同日重跑冗余。

    同一天可能存在多行（多次运行/中断），仅保留 updated_at 最大的一行，
    即该交易日最终完成写入的组合快照。
    """
    # 用日期下界 00:00:00 与上界 23:59:59 覆盖整个自然日
    resp = client.table("portfolio") \
        .select("id,config_id,trading_date,cashflow,total_assets,positions,updated_at") \
        .eq("config_id", config_id) \
        .gte("trading_date", f"{start}T00:00:00") \
        .lte("trading_date", f"{end}T23:59:59") \
        .order("trading_date") \
        .execute()
    raw = pd.DataFrame(resp.data)
    if raw.empty:
        raise RuntimeError(f"窗口 {start} ~ {end} 内无组合记录")

    raw["trade_date"] = pd.to_datetime(raw["trading_date"], utc=True).dt.tz_localize(None).dt.normalize()
    raw["updated_at"] = pd.to_datetime(raw["updated_at"], utc=True)
    raw["cashflow"] = raw["cashflow"].astype(float)
    raw["total_assets"] = raw["total_assets"].astype(float)

    # 每个交易日保留 updated_at 最新的一条
    chain = (
        raw.sort_values(["trade_date", "updated_at"])
           .groupby("trade_date", as_index=False)
           .last()
           .sort_values("trade_date")
           .reset_index(drop=True)
    )
    print(f"  组合快照: 原始 {len(raw)} 行 -> 去重后 {len(chain)} 个交易日")
    return chain


def fetch_decisions(client, portfolio_ids: List[str]) -> pd.DataFrame:
    """拉取指定组合的所有交易决策。"""
    resp = client.table("decision") \
        .select("portfolio_id,trading_date,ticker,action,shares,price,justification") \
        .in_("portfolio_id", portfolio_ids) \
        .order("trading_date") \
        .execute()
    df = pd.DataFrame(resp.data)
    if df.empty:
        raise RuntimeError("未找到任何决策记录")
    df["ticker"] = df["ticker"].astype(str)
    df["price"] = df["price"].astype(float)
    df["shares"] = df["shares"].astype(int)
    df["trade_date"] = pd.to_datetime(df["trading_date"], utc=True).dt.tz_localize(None).dt.normalize()
    return df


def fetch_signals(client, portfolio_ids: List[str], pid_to_date: Dict[str, pd.Timestamp]) -> pd.DataFrame:
    """拉取所有分析师信号（signal 表无 trading_date，通过 portfolio_id 映射日期）。"""
    resp = client.table("signal") \
        .select("portfolio_id,ticker,analyst,signal,justification") \
        .in_("portfolio_id", portfolio_ids) \
        .execute()
    df = pd.DataFrame(resp.data)
    if df.empty:
        raise RuntimeError("未找到任何信号记录")
    df["ticker"] = df["ticker"].astype(str)
    df["date"] = df["portfolio_id"].map(pid_to_date)
    return df


# ===========================================================================
# 2. 数据转换：补充次日收益、价格矩阵
# ===========================================================================
def build_price_matrix(decisions: pd.DataFrame) -> pd.DataFrame:
    """从决策记录提取每日每只股票的成交价 -> 价格矩阵（date × ticker）。"""
    return decisions.pivot_table(
        index="trade_date", columns="ticker", values="price", aggfunc="last"
    ).sort_index()


def attach_next_day_return(decisions: pd.DataFrame, price_matrix: pd.DataFrame) -> pd.DataFrame:
    """为每笔交易计算次日收益，用于胜率 WR（论文公式4）。

    次日收益 = 该股票下一交易日成交价 / 当日成交价 - 1。
    窗口最后一日无次日价格，记为 NaN（不参与主口径胜率）。
    """
    dates = list(price_matrix.index)
    df = decisions.copy()
    next_ret = []
    for _, row in df.iterrows():
        i = dates.index(row["trade_date"])
        if i + 1 < len(dates):
            nxt = price_matrix.iloc[i + 1].get(row["ticker"])
            next_ret.append(float(nxt) / row["price"] - 1 if pd.notna(nxt) and row["price"] > 0 else np.nan)
        else:
            next_ret.append(np.nan)
    df["next_day_return"] = next_ret
    return df


def to_trades_frame(decisions: pd.DataFrame) -> pd.DataFrame:
    """转换为 statistics.compute_wr_with_ci 所需的字段格式。"""
    return pd.DataFrame({
        "date": decisions["trade_date"].dt.strftime("%Y-%m-%d"),
        "code": decisions["ticker"],
        "action": decisions["action"],
        "shares": decisions["shares"],
        "price": decisions["price"],
        "next_day_return": decisions["next_day_return"],
    })


# ===========================================================================
# 3. 市场基准（沪深300）
# ===========================================================================
def fetch_market_returns(dates: List[pd.Timestamp], price_matrix: pd.DataFrame) -> np.ndarray:
    """获取沪深300在窗口内的日收益率；失败则回退为 5 只股票等权收益。"""
    try:
        import akshare as ak
        idx = ak.index_zh_hist_em(
            symbol=BENCHMARK_INDEX, period="daily",
            start_date=(dates[0] - pd.Timedelta(days=10)).strftime("%Y%m%d"),
            end_date=dates[-1].strftime("%Y%m%d"),
        )
        if idx is not None and not idx.empty:
            idx = idx.rename(columns={"日期": "date", "收盘": "close"})
            idx["date"] = pd.to_datetime(idx["date"])
            win = idx[idx["date"].isin(dates)].sort_values("date")
            if len(win) >= 2:
                close = win["close"].values.astype(float)
                print(f"  基准: 沪深300 实际指数（{len(win)} 个交易日）")
                return np.diff(close) / close[:-1]
    except Exception as e:
        print(f"  [WARN] 沪深300 获取失败，回退等权代理: {e}")

    eq = price_matrix.pct_change().dropna().mean(axis=1).values
    print("  基准: 5 只股票等权收益（代理）")
    return eq


# ===========================================================================
# 4. 三层数据校验
# ===========================================================================
def validate_assets_identity(chain: pd.DataFrame) -> List[Dict]:
    """校验 A：total_assets == cashflow + Σ position.value。"""
    rows = []
    for _, r in chain.iterrows():
        pos_val = sum(float(v.get("value", 0.0)) for v in (r["positions"] or {}).values())
        diff = abs(r["total_assets"] - (r["cashflow"] + pos_val))
        rows.append({
            "check": "A_资产恒等式",
            "trade_date": r["trade_date"].strftime("%Y-%m-%d"),
            "detail": f"total_assets={r['total_assets']:.2f} vs cash+pos={r['cashflow'] + pos_val:.2f}",
            "abs_diff": round(diff, 4),
            "passed": diff <= 0.01,
        })
    return rows


def validate_positions_consistency(chain: pd.DataFrame, decisions: pd.DataFrame) -> List[Dict]:
    """校验 B：第 d 日持仓 = 前 d 日 Buy/Sell 的累积净额。"""
    rows = []
    dates = list(chain["trade_date"])
    for d in dates:
        snap = chain[chain["trade_date"] == d].iloc[0]
        db_pos = {k: int(v.get("shares", 0)) for k, v in (snap["positions"] or {}).items()}
        cum = decisions[decisions["trade_date"] <= d].copy()
        cum["delta"] = np.where(cum["action"] == "Buy", cum["shares"],
                                np.where(cum["action"] == "Sell", -cum["shares"], 0))
        expect = cum.groupby("ticker")["delta"].sum().to_dict()

        for t in sorted(set(db_pos) | set(expect)):
            ok = db_pos.get(t, 0) == int(expect.get(t, 0))
            rows.append({
                "check": "B_持仓累计一致性",
                "trade_date": d.strftime("%Y-%m-%d"),
                "detail": f"{t}: db={db_pos.get(t, 0)} vs cumdecisions={int(expect.get(t, 0))}",
                "abs_diff": abs(db_pos.get(t, 0) - int(expect.get(t, 0))),
                "passed": ok,
            })
    return rows


def validate_decision_completeness(chain: pd.DataFrame, decisions: pd.DataFrame, tickers: List[str]) -> List[Dict]:
    """校验 C：每个交易日决策数 == 配置 ticker 数，且 ticker 集合匹配。"""
    rows = []
    for _, r in chain.iterrows():
        d = r["trade_date"]
        day = decisions[decisions["trade_date"] == d]
        got = set(day["ticker"])
        want = set(tickers)
        ok = len(day) == len(tickers) and got == want
        rows.append({
            "check": "C_决策完整性",
            "trade_date": d.strftime("%Y-%m-%d"),
            "detail": f"decisions={len(day)}/{len(tickers)}, missing={sorted(want - got)}, extra={sorted(got - want)}",
            "abs_diff": abs(len(day) - len(tickers)),
            "passed": ok,
        })
    return rows


# ===========================================================================
# 5. 统计计算（复用 statistics.py 原语）
# ===========================================================================
def compute_metrics(chain: pd.DataFrame, decisions: pd.DataFrame,
                    market_returns: np.ndarray) -> Dict:
    """计算论文 Appendix D 全部 7 项指标 + CI + p 值 + 效应量。"""
    daily_values = chain["total_assets"].values.astype(float)
    strat_rets = np.diff(daily_values) / daily_values[:-1]
    dates = list(chain["trade_date"])

    # 1) CR / 3) SR / 4) MDD
    cr = stats_mod.compute_cr_with_ci(daily_values)
    sr = stats_mod.compute_sr_with_ci(daily_values, RISK_FREE_RATE)
    mdd = stats_mod.compute_mdd_with_ci(daily_values)

    # 5) WR —— 主口径：仅统计有次日收益的交易（论文公式4）
    closed = decisions[decisions["next_day_return"].notna()]
    wr = stats_mod.compute_wr_with_ci(to_trades_frame(closed))
    wr_all = stats_mod.compute_wr_with_ci(to_trades_frame(decisions))

    # 2) CR_bnh —— 首日等权买入持有
    price_matrix = build_price_matrix(decisions)
    first, last = price_matrix.iloc[0], price_matrix.iloc[-1]
    cr_bnh = stats_mod.compute_cr_bnh(first.to_dict(), last.to_dict(), INITIAL_CASH)

    # 6) Beta / 7) Alpha —— CAPM 回归
    capm = stats_mod.compute_capm_stats(strat_rets, market_returns, RISK_FREE_RATE)

    # 补充：信息比率与主动收益
    ir = stats_mod.compute_information_ratio(strat_rets, market_returns)

    return {
        "daily_values": daily_values,
        "strat_rets": strat_rets,
        "price_matrix": price_matrix,
        "core": {
            "CR": cr,
            "CR_bnh": {"metric": "CR_bnh", "estimate": round(cr_bnh, 4)},
            "SR": sr,
            "MDD": mdd,
            "WR": wr,
            "Beta": {"metric": "Beta", "estimate": capm["beta"],
                     "ci_lower": capm["beta_ci_lower"], "ci_upper": capm["beta_ci_upper"],
                     "p_value": capm["beta_p"]},
            "Alpha": {"metric": "Alpha", "estimate": capm["alpha_annualized_pct"],
                      "ci_lower": round(capm["alpha_ci_lower"] * 252 * 100, 4),
                      "ci_upper": round(capm["alpha_ci_upper"] * 252 * 100, 4),
                      "p_value": capm["alpha_p"]},
        },
        "wr_all": wr_all,
        "capm": capm,
        "ir": ir,
    }


# ===========================================================================
# 6. 本地落盘
# ===========================================================================
def export_raw(chain: pd.DataFrame, decisions: pd.DataFrame, signals: pd.DataFrame,
               price_matrix: pd.DataFrame) -> Dict[str, str]:
    """保存原始实验记录（组合 / 持仓 / 价格 / 决策 / 信号）为 CSV。"""
    paths = {}
    enc = "utf-8-sig"

    # portfolio_history.csv
    hist = chain[["trade_date", "id", "cashflow", "total_assets"]].copy()
    hist["trade_date"] = hist["trade_date"].dt.strftime("%Y-%m-%d")
    hist["daily_return_pct"] = hist["total_assets"].pct_change() * 100
    hist.columns = ["trade_date", "portfolio_id", "cashflow", "total_assets", "daily_return_pct"]
    p = os.path.join(OUTPUT_DIR, "portfolio_history.csv")
    hist.to_csv(p, index=False, encoding=enc); paths["portfolio_history"] = p

    # positions_history.csv（长表）
    rows = []
    for _, r in chain.iterrows():
        for t, v in (r["positions"] or {}).items():
            rows.append({
                "trade_date": r["trade_date"].strftime("%Y-%m-%d"),
                "ticker": t,
                "shares": int(v.get("shares", 0)),
                "value": float(v.get("value", 0.0)),
                "weight_pct": round(float(v.get("value", 0.0)) / r["total_assets"] * 100, 4)
                              if r["total_assets"] else np.nan,
            })
    p = os.path.join(OUTPUT_DIR, "positions_history.csv")
    pd.DataFrame(rows).to_csv(p, index=False, encoding=enc); paths["positions_history"] = p

    # price_matrix.csv
    pm = price_matrix.copy(); pm.index = pm.index.strftime("%Y-%m-%d")
    p = os.path.join(OUTPUT_DIR, "price_matrix.csv")
    pm.to_csv(p, encoding=enc); paths["price_matrix"] = p

    # decisions.csv
    dec = decisions.copy()
    dec["trade_date"] = dec["trade_date"].dt.strftime("%Y-%m-%d")
    dec[["trade_date", "ticker", "action", "shares", "price", "next_day_return", "justification"]] \
        .to_csv(os.path.join(OUTPUT_DIR, "decisions.csv"), index=False, encoding=enc)
    paths["decisions"] = os.path.join(OUTPUT_DIR, "decisions.csv")

    # signals.csv
    sig = signals.copy()
    sig["date"] = pd.to_datetime(sig["date"]).dt.strftime("%Y-%m-%d")
    sig[["date", "ticker", "analyst", "signal", "justification"]].sort_values(["date", "ticker", "analyst"]) \
        .to_csv(os.path.join(OUTPUT_DIR, "signals.csv"), index=False, encoding=enc)
    paths["signals"] = os.path.join(OUTPUT_DIR, "signals.csv")
    return paths


def export_statistics(result: Dict, validation: pd.DataFrame) -> Dict[str, str]:
    """保存统计结果表（论文 Table 3 / Table 4 / 稳健性）。"""
    paths = {}
    enc = "utf-8-sig"

    # Table 3 核心指标
    rows = []
    for key, v in result["core"].items():
        rows.append({
            "指标": v["metric"],
            "点估计": v["estimate"],
            "95%CI下界": v.get("ci_lower"),
            "95%CI上界": v.get("ci_upper"),
            "p值": v.get("p_value"),
            "检验方法": v.get("test"),
            "效应量": v.get("effect_size"),
        })
    # 胜率敏感性（含无次日收益的交易）
    rows.append({
        "指标": "WR(含未平仓)", "点估计": result["wr_all"]["estimate"],
        "95%CI下界": result["wr_all"]["ci_lower"], "95%CI上界": result["wr_all"]["ci_upper"],
        "p值": result["wr_all"]["p_value"], "检验方法": "Binomial test (H0: win rate = 50%)",
        "效应量": None,
    })
    p = os.path.join(OUTPUT_DIR, "table3_core_metrics.csv")
    pd.DataFrame(rows).to_csv(p, index=False, encoding=enc); paths["table3"] = p

    # Table 4 CAPM
    c = result["capm"]
    t4 = pd.DataFrame([
        {"参数": "Alpha (α, 日)", "估计值": c["alpha"], "标准误": c["alpha_se"],
         "t值": c["alpha_t"], "p值": c["alpha_p"], "95%CI": f"[{c['alpha_ci_lower']}, {c['alpha_ci_upper']}]"},
        {"参数": "Alpha (年化%)", "估计值": c["alpha_annualized_pct"], "标准误": None,
         "t值": None, "p值": c["alpha_p"], "95%CI": None},
        {"参数": "Beta (β)", "估计值": c["beta"], "标准误": c["beta_se"],
         "t值": c["beta_t"], "p值": c["beta_p"], "95%CI": f"[{c['beta_ci_lower']}, {c['beta_ci_upper']}]"},
        {"参数": "R²", "估计值": c["r_squared"], "标准误": None, "t值": None, "p值": None, "95%CI": None},
        {"参数": "观测数 N", "估计值": c["n_obs"], "标准误": None, "t值": None, "p值": None, "95%CI": None},
        {"参数": "信息比率 IR", "估计值": result["ir"]["information_ratio"], "标准误": None,
         "t值": None, "p值": None, "95%CI": f"跟踪误差={result['ir']['tracking_error_annualized']}"},
    ])
    p = os.path.join(OUTPUT_DIR, "table4_capm_regression.csv")
    t4.to_csv(p, index=False, encoding=enc); paths["table4"] = p

    # 稳健性（复用 statistics.run_robustness）
    rob = stats_mod.run_robustness(result["daily_values"], result["market_returns"], result["strat_rets"])
    for name in ["risk_free_rate", "benchmark", "window"]:
        p = os.path.join(OUTPUT_DIR, f"table5_robustness_{name}.csv")
        rob[name].to_csv(p, index=False, encoding=enc); paths[f"robustness_{name}"] = p

    # 校验报告
    p = os.path.join(OUTPUT_DIR, "validation_report.csv")
    validation.to_csv(p, index=False, encoding=enc); paths["validation"] = p

    # 信号汇总
    return paths


def export_signals_summary(signals: pd.DataFrame) -> str:
    """分析师信号分布汇总（对应论文 Figure 9 / Table 2）。"""
    s = signals.groupby(["analyst", "signal"]).size().unstack(fill_value=0)
    s["total"] = s.sum(axis=1)
    p = os.path.join(OUTPUT_DIR, "signals_summary.csv")
    s.to_csv(p, encoding="utf-8-sig")
    return p


# ===========================================================================
# 7. 主流程
# ===========================================================================
def main():
    global OUTPUT_DIR
    ap = argparse.ArgumentParser(description="基于 Supabase 实验数据的统计结果生成")
    ap.add_argument("--exp-name", default="minimax-test")
    ap.add_argument("--start", default="2026-11-10")
    ap.add_argument("--end", default="2026-11-16")
    ap.add_argument("--output-dir", default=OUTPUT_DIR)
    args = ap.parse_args()

    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 72)
    print(f"DeepFund 实验统计（Supabase 数据源）  {args.start} ~ {args.end}")
    print("=" * 72)

    # --- Step 1: 连接与拉取 ---
    print("\n[1] 连接 Supabase 并拉取实验记录")
    client = get_client()
    config_id = resolve_config_id(client, args.exp_name)
    tickers = client.table("config").select("tickers").eq("id", config_id).execute().data[0]["tickers"]

    chain = fetch_portfolio_chain(client, config_id, args.start, args.end)
    pids = chain["id"].tolist()
    pid_to_date = dict(zip(chain["id"], chain["trade_date"]))
    decisions = fetch_decisions(client, pids)
    signals = fetch_signals(client, pids, pid_to_date)
    print(f"  决策记录: {len(decisions)} 条 | 信号记录: {len(signals)} 条")

    # --- Step 2: 数据转换 ---
    print("\n[2] 数据转换（价格矩阵 / 次日收益）")
    price_matrix = build_price_matrix(decisions)
    decisions = attach_next_day_return(decisions, price_matrix)
    print(f"  价格矩阵: {price_matrix.shape[0]} 交易日 × {price_matrix.shape[1]} 标的")

    # --- Step 3: 市场基准 ---
    print("\n[3] 获取市场基准收益")
    dates = list(chain["trade_date"])
    market_returns = fetch_market_returns(dates, price_matrix)

    # --- Step 4: 统计计算 ---
    print("\n[4] 计算论文 Appendix D 核心指标（CI 由 Bootstrap B=10000 生成）")
    result = compute_metrics(chain, decisions, market_returns)
    result["market_returns"] = market_returns

    print(f"\n  {'指标':<10}{'点估计':>12}{'95%CI':>28}{'p值':>10}")
    for key, v in result["core"].items():
        ci = f"[{v.get('ci_lower')}, {v.get('ci_upper')}]" if v.get("ci_lower") is not None else "-"
        p = v.get("p_value")
        print(f"  {v['metric']:<10}{v['estimate']:>12}{ci:>28}{(p if p is not None else '-'):>10}")
    print(f"  {'WR(全量)':<10}{result['wr_all']['estimate']:>12}"
          f"{'[' + str(result['wr_all']['ci_lower']) + ', ' + str(result['wr_all']['ci_upper']) + ']':>28}"
          f"{result['wr_all']['p_value']:>10}")

    # --- Step 5: 校验 ---
    print("\n[5] 三层数据校验")
    v_rows = (validate_assets_identity(chain)
              + validate_positions_consistency(chain, decisions)
              + validate_decision_completeness(chain, decisions, tickers))
    validation = pd.DataFrame(v_rows)
    for ck, g in validation.groupby("check", sort=True):
        ok = int(g["passed"].sum())
        print(f"  [{'PASS' if ok == len(g) else 'FAIL'}] {ck}: {ok}/{len(g)} 通过")
    print(f"  整体校验: {'ALL PASSED' if validation['passed'].all() else 'SOME FAILED'}")

    # --- Step 6: 落盘 ---
    print("\n[6] 本地落盘 CSV")
    paths = export_raw(chain, decisions, signals, price_matrix)
    paths.update(export_statistics(result, validation))
    paths["signals_summary"] = export_signals_summary(signals)
    for k, v in paths.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 72)
    print(f"完成。输出目录: {OUTPUT_DIR}")
    print("=" * 72)
    return result, validation


if __name__ == "__main__":
    main()
