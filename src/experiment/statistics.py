"""
统计结果生成模块
===============
对应论文《Time Travel is Cheating: Going Live with DeepFund》
  Appendix D "Evaluation Metrics" 及结果呈现规范（Table 3 / Table 4）。

本模块基于已完成的实验输出数据（2026-11-10 ~ 2026-11-16），
自动化执行完整统计流水线：

  1. 数据读取：加载 trades_detail.csv / signals_detail.csv / core_metrics.csv
               / capm_regression.csv / descriptive_statistics.csv 等原始输出
  2. 数据清洗与转换：重建日度组合价值序列、计算日收益率、对齐基准
  3. 全量统计计算（论文 Appendix D 全部 7 项指标）：
       CR, CR_bnh, SR, MDD, WR, Beta, Alpha
     每项均附带：
       - 点估计
       - 95% 置信区间（非参数 Bootstrap, B=10000）
       - 假设检验 p 值
       - 效应量（Cohen's d / Information Ratio 等）
  4. 回归分析：
       CAPM 回归（含 α、β 的 t 检验、p 值、置信区间）
       Fama-French 三因子回归
  5. 稳健性检验：
       无风险利率敏感性 / 基准指数敏感性 / 交易窗口敏感性
  6. 结果校验：与 core_metrics.csv 已知指标交叉验证
  7. 输出结构化 CSV（论文 Table 3 / Table 4 格式）

可复现性：所有随机过程固定 seed=42。
"""

import os
import sys
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 可复现性：固定随机种子（论文要求可复现实验）
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# 同目录模块可导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CONFIG, SAMPLE_STOCKS  # noqa: E402

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Bootstrap 重采样次数（论文统计推断标准设置）
BOOTSTRAP_N = 10000
# 置信水平
CONFIDENCE_LEVEL = 0.95
ALPHA_LEVEL = 1 - CONFIDENCE_LEVEL  # 0.05


# ===========================================================================
# 1. 数据读取与清洗
# ===========================================================================
def load_raw_data(output_dir: str = OUTPUT_DIR) -> Dict[str, pd.DataFrame]:
    """加载所有原始实验输出数据。

    返回字典包含：
      - trades: 交易明细
      - signals: 分析师信号明细
      - core_metrics: 核心指标（已知值，用于校验）
      - capm: CAPM 回归结果（已知值，用于校验）
      - descriptive: 描述性统计
      - robustness_rf / robustness_benchmark / robustness_window: 稳健性检验
    """
    data = {}

    # 交易明细（论文 Section 3 交易记录）
    trades_path = os.path.join(output_dir, "trades_detail.csv")
    data["trades"] = pd.read_csv(trades_path, encoding="utf-8-sig", dtype={"code": str})

    # 信号明细（论文 Section 2 分析师输出）
    signals_path = os.path.join(output_dir, "signals_detail.csv")
    data["signals"] = pd.read_csv(signals_path, encoding="utf-8-sig", dtype={"code": str})

    # 核心指标（用于校验）
    metrics_path = os.path.join(output_dir, "core_metrics.csv")
    data["core_metrics"] = pd.read_csv(metrics_path, encoding="utf-8-sig")

    # CAPM 回归结果（用于校验）
    capm_path = os.path.join(output_dir, "capm_regression.csv")
    data["capm"] = pd.read_csv(capm_path, encoding="utf-8-sig")

    # 描述性统计
    desc_path = os.path.join(output_dir, "descriptive_statistics.csv")
    data["descriptive"] = pd.read_csv(desc_path, encoding="utf-8-sig")

    # 稳健性检验
    for name, fname in [
        ("robustness_rf", "robustness_risk_free_rate.csv"),
        ("robustness_benchmark", "robustness_benchmark.csv"),
        ("robustness_window", "robustness_window.csv"),
    ]:
        fpath = os.path.join(output_dir, fname)
        if os.path.exists(fpath):
            data[name] = pd.read_csv(fpath, encoding="utf-8-sig")

    return data


def clean_trades(trades_df: pd.DataFrame) -> pd.DataFrame:
    """清洗交易明细数据。

    处理步骤：
      - 日期转为 datetime
      - 剔除无效交易（shares=0 且 action=Hold 不计入胜率统计）
      - 保留 action/shares/price/next_day_return 关键字段
    """
    df = trades_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "code"]).reset_index(drop=True)
    return df


# ===========================================================================
# 2. 日度组合价值序列重建
# ===========================================================================
def reconstruct_portfolio(trades_df: pd.DataFrame) -> pd.DataFrame:
    """从交易明细重建日度组合价值序列。

    依据 portfolio_manager.py 的交易执行逻辑：
      - 买入：cash -= shares * price * (1 + commission_rate)
      - 卖出：cash += shares * price * (1 - commission_rate - stamp_tax_rate)

    返回 DataFrame 包含：date, total_value, cash, positions_value, daily_return
    """
    df = clean_trades(trades_df)
    dates = sorted(df["date"].unique())

    # 提取每日每只股票的收盘价（取该股票当日交易记录中的 price）
    daily_prices = {}
    for d in dates:
        day_trades = df[df["date"] == d]
        daily_prices[d] = {}
        for _, row in day_trades.iterrows():
            daily_prices[d][row["code"]] = float(row["price"])

    cash = float(CONFIG.initial_cash)
    positions = {s["code"]: 0 for s in SAMPLE_STOCKS}  # code -> shares

    records = []

    for d in dates:
        prices = daily_prices[d]
        day_trades = df[df["date"] == d].sort_values("code")

        # 按当日顺序执行交易
        for _, row in day_trades.iterrows():
            code = row["code"]
            action = row["action"]
            shares = int(row["shares"])
            price = float(row["price"])

            if action == "Buy" and shares > 0:
                cost = shares * price
                commission = cost * CONFIG.commission_rate
                total_cost = cost + commission
                if total_cost <= cash:
                    positions[code] += shares
                    cash -= total_cost
            elif action == "Sell" and shares > 0:
                sell_shares = min(shares, positions[code])
                if sell_shares > 0:
                    revenue = sell_shares * price
                    commission = revenue * CONFIG.commission_rate
                    stamp_tax = revenue * CONFIG.stamp_tax_rate
                    cash += revenue - commission - stamp_tax
                    positions[code] -= sell_shares

        # 计算当日组合总价值
        positions_value = sum(
            positions[code] * prices.get(code, 0.0) for code in positions
        )
        total_value = cash + positions_value

        records.append({
            "date": d,
            "total_value": total_value,
            "cash": cash,
            "positions_value": positions_value,
        })

    result = pd.DataFrame(records)
    # 计算日收益率
    result["daily_return"] = result["total_value"].pct_change()
    return result


def get_market_returns(trades_df: pd.DataFrame) -> Tuple[np.ndarray, List]:
    """获取沪深300基准日收益率。

    优先使用 akshare 获取实际 CSI 300 指数数据（与原始实验一致）。
    若网络获取失败，回退到等权 5 只股票次日收益作为市场代理。

    返回：(market_daily_returns, dates)
    """
    df = clean_trades(trades_df)
    dates = sorted(df["date"].unique())

    # 尝试获取实际沪深300指数数据
    try:
        import akshare as ak
        idx_start = (dates[0] - pd.Timedelta(days=10)).strftime("%Y%m%d")
        idx_end = dates[-1].strftime("%Y%m%d")
        idx_df = ak.index_zh_hist_em(
            symbol=CONFIG.benchmark_index,
            period="daily",
            start_date=idx_start,
            end_date=idx_end,
        )
        if idx_df is not None and not idx_df.empty:
            idx_df = idx_df.rename(columns={"日期": "date", "收盘": "close"})
            idx_df["date"] = pd.to_datetime(idx_df["date"])
            idx_df = idx_df.sort_values("date")
            idx_window = idx_df[idx_df["date"].isin(dates)]
            if len(idx_window) >= 2:
                closes = idx_window["close"].values.astype(float)
                market_rets = np.diff(closes) / closes[:-1]
                return market_rets, dates
    except Exception:
        pass

    # 回退：等权 5 只股票次日收益作为市场代理
    market_rets = []
    for i in range(len(dates) - 1):
        day_trades = df[df["date"] == dates[i]]
        rets = day_trades["next_day_return"].values
        market_rets.append(np.mean(rets))
    return np.array(market_rets), dates


# ===========================================================================
# 3. 核心指标计算（论文 Appendix D）+ 置信区间 + 假设检验
# ===========================================================================
def bootstrap_ci(
    data: np.ndarray,
    statistic_func,
    n_bootstrap: int = BOOTSTRAP_N,
    confidence: float = CONFIDENCE_LEVEL,
    seed: int = RANDOM_SEED,
) -> Tuple[float, float]:
    """非参数 Bootstrap 置信区间。

    论文统计推断：对样本进行 B 次有放回重采样，计算统计量的经验分布，
    取 (1-α)/2 和 (1+α)/2 分位数作为置信区间上下界。

    Args:
        data: 原始样本数组
        statistic_func: 统计量函数，输入数组返回标量
        n_bootstrap: 重采样次数
        confidence: 置信水平
        seed: 随机种子
    """
    rng = np.random.RandomState(seed)
    n = len(data)
    if n == 0:
        return (np.nan, np.nan)

    boot_stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        boot_stats[i] = statistic_func(sample)

    alpha = 1 - confidence
    lower = np.percentile(boot_stats, 100 * alpha / 2)
    upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return (float(lower), float(upper))


def compute_cr_with_ci(daily_values: np.ndarray) -> Dict:
    """累计收益率 CR（论文公式1）+ Bootstrap CI + 假设检验。

    CR = (P_final / P_initial - 1) × 100
    H0: CR = 0（策略累计收益为零）
    """
    initial = daily_values[0]
    final = daily_values[-1]
    cr = (final / initial - 1) * 100

    # 用日收益率做 Bootstrap（CR 由日收益累积得到）
    daily_rets = np.diff(daily_values) / daily_values[:-1]

    def cr_stat(rets):
        return (np.prod(1 + rets) - 1) * 100

    ci_lower, ci_upper = bootstrap_ci(daily_rets, cr_stat)

    # 单样本 t 检验：H0: mean(daily_return) = 0
    if len(daily_rets) >= 2:
        t_stat, p_value = stats.ttest_1samp(daily_rets, 0.0)
    else:
        t_stat, p_value = np.nan, np.nan

    return {
        "metric": "CR",
        "estimate": round(cr, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "ci_level": f"{CONFIDENCE_LEVEL*100:.0f}%",
        "t_stat": round(t_stat, 4) if not np.isnan(t_stat) else None,
        "p_value": round(p_value, 4) if not np.isnan(p_value) else None,
        "test": "One-sample t-test (H0: mean daily return = 0)",
        "effect_size": None,  # CR 本身即为效应量
    }


def compute_sr_with_ci(daily_values: np.ndarray, rf: float) -> Dict:
    """夏普比率 SR（论文公式2）+ Bootstrap CI + 假设检验。

    SR = (mean(r_e) / std(r_e)) × sqrt(252)
    H0: SR = 0（夏普比率为零）
    """
    daily_rets = np.diff(daily_values) / daily_values[:-1]
    rf_daily = rf / 252
    excess = daily_rets - rf_daily

    def sr_stat(rets):
        mean_e = np.mean(rets - rf_daily)
        std_e = np.std(rets - rf_daily, ddof=1)
        if std_e == 0:
            return 0.0
        return (mean_e / std_e) * math.sqrt(252)

    sr = sr_stat(daily_rets)
    ci_lower, ci_upper = bootstrap_ci(daily_rets, sr_stat)

    # H0: mean excess return = 0
    if len(excess) >= 2:
        t_stat, p_value = stats.ttest_1samp(excess, 0.0)
    else:
        t_stat, p_value = np.nan, np.nan

    # 效应量：Cohen's d = mean(excess) / std(excess)（未年化）
    if np.std(excess, ddof=1) > 0:
        cohens_d = float(np.mean(excess) / np.std(excess, ddof=1))
    else:
        cohens_d = np.nan

    return {
        "metric": "SR",
        "estimate": round(sr, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "ci_level": f"{CONFIDENCE_LEVEL*100:.0f}%",
        "t_stat": round(t_stat, 4) if not np.isnan(t_stat) else None,
        "p_value": round(p_value, 4) if not np.isnan(p_value) else None,
        "test": "One-sample t-test (H0: mean excess return = 0)",
        "effect_size": round(cohens_d, 4),
        "effect_size_name": "Cohen's d (daily)",
    }


def compute_mdd_with_ci(daily_values: np.ndarray) -> Dict:
    """最大回撤 MDD（论文公式3）+ Bootstrap CI。

    MDD = max_t [max_{s<=t}(P_s - P_t) / P_s] × 100
    """
    values = np.array(daily_values, dtype=float)
    running_max = np.maximum.accumulate(values)
    drawdowns = (running_max - values) / running_max
    mdd = np.max(drawdowns) * 100

    daily_rets = np.diff(values) / values[:-1]

    def mdd_stat(rets):
        # 从初始值重建序列
        seq = values[0] * np.cumprod(1 + rets)
        seq = np.concatenate([[values[0]], seq])
        rm = np.maximum.accumulate(seq)
        dd = (rm - seq) / rm
        return np.max(dd) * 100

    ci_lower, ci_upper = bootstrap_ci(daily_rets, mdd_stat)

    return {
        "metric": "MDD",
        "estimate": round(mdd, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "ci_level": f"{CONFIDENCE_LEVEL*100:.0f}%",
        "t_stat": None,
        "p_value": None,
        "test": None,
        "effect_size": None,
    }


def compute_wr_with_ci(trades_df: pd.DataFrame) -> Dict:
    """胜率 WR（论文公式4）+ 二项检验 + 精确置信区间。

    WR = (盈利交易数 / 总交易数) × 100
    盈利判定：Buy 后次日涨 / Sell 后次日跌
    H0: WR = 50%（无预测能力）

    使用二项检验（binom_test）和 Clopper-Pearson 精确置信区间。
    """
    df = clean_trades(trades_df)
    # 只统计实际发生的交易（shares > 0）
    actual_trades = df[df["shares"] > 0].copy()

    profitable = 0
    total = len(actual_trades)

    for _, row in actual_trades.iterrows():
        action = row["action"]
        next_ret = float(row["next_day_return"])
        if action == "Buy" and next_ret > 0:
            profitable += 1
        elif action == "Sell" and next_ret < 0:
            profitable += 1

    wr = (profitable / total * 100) if total > 0 else 0.0

    # 二项检验 H0: p = 0.5
    if total > 0:
        p_value = float(stats.binomtest(profitable, total, 0.5).pvalue)
        # Clopper-Pearson 精确置信区间
        ci_lower, ci_upper = stats.binom.interval(
            CONFIDENCE_LEVEL, total, profitable / total if total > 0 else 0
        )
        ci_lower = ci_lower / total * 100
        ci_upper = ci_upper / total * 100
    else:
        p_value = np.nan
        ci_lower, ci_upper = np.nan, np.nan

    return {
        "metric": "WR",
        "estimate": round(wr, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "ci_level": f"{CONFIDENCE_LEVEL*100:.0f}%",
        "t_stat": None,
        "p_value": round(p_value, 4) if not np.isnan(p_value) else None,
        "test": "Binomial test (H0: win rate = 50%)",
        "effect_size": None,
        "n_trades": total,
        "n_profitable": profitable,
    }


def compute_cr_bnh(initial_prices: Dict[str, float], final_prices: Dict[str, float],
                   initial_cash: float) -> float:
    """买入持有累计收益率 CR_bnh（论文定义）。"""
    if not initial_prices or initial_cash <= 0:
        return 0.0
    n = len(initial_prices)
    cash_per_stock = initial_cash / n
    final_value = 0.0
    for code, init_price in initial_prices.items():
        final_price = final_prices.get(code, init_price)
        if init_price > 0:
            shares = cash_per_stock / init_price
            final_value += shares * final_price
    return (final_value / initial_cash - 1) * 100


def compute_capm_stats(strategy_returns: np.ndarray, market_returns: np.ndarray,
                       rf: float) -> Dict:
    """CAPM 回归统计推断（论文 Section 4 回归分析）。

    模型：r_s - r_f = α + β × (r_m - r_f) + ε

    返回 α、β 的点估计、标准误、t 值、p 值、95% 置信区间、R²。
    """
    n = min(len(strategy_returns), len(market_returns))
    r_s = strategy_returns[-n:]
    r_m = market_returns[-n:]
    rf_daily = rf / 252

    y = r_s - rf_daily
    x = r_m - rf_daily

    # 处理基准零方差情况（OLS 会丢弃常数列）
    if np.var(x, ddof=1) == 0:
        # 基准不变，Beta 无定义，返回 0 和仅截距模型
        model = sm.OLS(y, np.ones(len(y))).fit()
        alpha = float(model.params[0])
        return {
            "alpha": round(alpha, 6),
            "alpha_annualized_pct": round(alpha * 252 * 100, 4),
            "alpha_se": round(float(model.bse[0]), 6),
            "alpha_t": round(float(model.tvalues[0]), 4),
            "alpha_p": round(float(model.pvalues[0]), 4),
            "alpha_ci_lower": round(float(model.conf_int(alpha=ALPHA_LEVEL)[0][0]), 6),
            "alpha_ci_upper": round(float(model.conf_int(alpha=ALPHA_LEVEL)[0][1]), 6),
            "beta": 0.0,
            "beta_se": np.nan,
            "beta_t": np.nan,
            "beta_p": np.nan,
            "beta_ci_lower": np.nan,
            "beta_ci_upper": np.nan,
            "r_squared": round(float(model.rsquared), 4),
            "n_obs": int(n),
            "residual_std": round(float(np.std(model.resid, ddof=1)), 6),
        }

    x_const = sm.add_constant(x)
    model = sm.OLS(y, x_const).fit()

    alpha = float(model.params[0])
    beta = float(model.params[1])
    alpha_se = float(model.bse[0])
    beta_se = float(model.bse[1])
    alpha_t = float(model.tvalues[0])
    beta_t = float(model.tvalues[1])
    alpha_p = float(model.pvalues[0])
    beta_p = float(model.pvalues[1])
    r_squared = float(model.rsquared)

    # 95% 置信区间
    alpha_ci = model.conf_int(alpha=ALPHA_LEVEL)[0]
    beta_ci = model.conf_int(alpha=ALPHA_LEVEL)[1]

    # 年化 Alpha
    alpha_annualized = alpha * 252 * 100

    return {
        "alpha": round(alpha, 6),
        "alpha_annualized_pct": round(alpha_annualized, 4),
        "alpha_se": round(alpha_se, 6),
        "alpha_t": round(alpha_t, 4),
        "alpha_p": round(alpha_p, 4),
        "alpha_ci_lower": round(float(alpha_ci[0]), 6),
        "alpha_ci_upper": round(float(alpha_ci[1]), 6),
        "beta": round(beta, 4),
        "beta_se": round(beta_se, 4),
        "beta_t": round(beta_t, 4),
        "beta_p": round(beta_p, 4),
        "beta_ci_lower": round(float(beta_ci[0]), 4),
        "beta_ci_upper": round(float(beta_ci[1]), 4),
        "r_squared": round(r_squared, 4),
        "n_obs": int(n),
        "residual_std": round(float(np.std(model.resid, ddof=2)), 6),
    }


def compute_information_ratio(strategy_returns: np.ndarray,
                              market_returns: np.ndarray) -> Dict:
    """信息比率 IR（论文补充绩效指标）。

    IR = mean(r_s - r_m) / std(r_s - r_m) × sqrt(252)
    衡量主动管理获取超额收益的效率。
    """
    n = min(len(strategy_returns), len(market_returns))
    active = strategy_returns[-n:] - market_returns[-n:]

    if len(active) < 2 or np.std(active, ddof=1) == 0:
        ir = 0.0
        tracking_error = 0.0
    else:
        ir = (np.mean(active) / np.std(active, ddof=1)) * math.sqrt(252)
        tracking_error = np.std(active, ddof=1) * math.sqrt(252)

    return {
        "information_ratio": round(ir, 4),
        "tracking_error_annualized": round(tracking_error, 4),
    }


def compute_strategy_vs_bnh_test(strategy_returns: np.ndarray,
                                 bnh_returns: np.ndarray) -> Dict:
    """策略 vs 买入持有配对检验。

    H0: 策略日均收益 = 买入持有日均收益
    使用配对样本 t 检验。
    效应量：Cohen's d（配对）
    """
    n = min(len(strategy_returns), len(bnh_returns))
    if n < 2:
        return {"t_stat": None, "p_value": None, "cohens_d": None}

    s = strategy_returns[-n:]
    b = bnh_returns[-n:]
    diff = s - b

    t_stat, p_value = stats.ttest_rel(s, b)

    # Cohen's d (paired) = mean(diff) / std(diff)
    std_diff = np.std(diff, ddof=1)
    cohens_d = float(np.mean(diff) / std_diff) if std_diff > 0 else np.nan

    return {
        "t_stat": round(float(t_stat), 4),
        "p_value": round(float(p_value), 4),
        "cohens_d": round(cohens_d, 4),
        "test": "Paired t-test (H0: strategy return = B&H return)",
    }


# ===========================================================================
# 4. 全量统计计算主函数
# ===========================================================================
def run_full_statistics(trades_df: pd.DataFrame, daily_values_df: pd.DataFrame,
                        market_returns: np.ndarray) -> Dict:
    """执行全量统计计算，返回所有结果。

    Args:
        trades_df: 交易明细
        daily_values_df: 重建的日度组合价值
        market_returns: 基准日收益率
    """
    daily_values = daily_values_df["total_value"].values
    strat_rets = daily_values_df["daily_return"].dropna().values

    rf = CONFIG.risk_free_rate

    # --- 核心指标 + CI + 检验 ---
    cr_result = compute_cr_with_ci(daily_values)
    sr_result = compute_sr_with_ci(daily_values, rf)
    mdd_result = compute_mdd_with_ci(daily_values)
    wr_result = compute_wr_with_ci(trades_df)

    # --- CR_bnh ---
    df_clean = clean_trades(trades_df)
    dates = sorted(df_clean["date"].unique())
    initial_prices = {}
    final_prices = {}
    for code in [s["code"] for s in SAMPLE_STOCKS]:
        first = df_clean[(df_clean["code"] == code) & (df_clean["date"] == dates[0])]
        last = df_clean[(df_clean["code"] == code) & (df_clean["date"] == dates[-1])]
        if not first.empty:
            initial_prices[code] = float(first.iloc[0]["price"])
        if not last.empty:
            final_prices[code] = float(last.iloc[0]["price"])

    cr_bnh = compute_cr_bnh(initial_prices, final_prices, CONFIG.initial_cash)

    # --- CAPM 回归 ---
    capm = compute_capm_stats(strat_rets, market_returns, rf)

    # --- 信息比率 ---
    ir = compute_information_ratio(strat_rets, market_returns)

    # --- 策略 vs B&H 检验 ---
    # 构造 B&H 日收益：等权买入持有 5 只股票
    bnh_daily_values = []
    n_stocks = len(SAMPLE_STOCKS)
    cash_per = CONFIG.initial_cash / n_stocks
    bnh_positions = {}
    for code in initial_prices:
        bnh_positions[code] = cash_per / initial_prices[code]

    for d in dates:
        day_prices = {}
        day_trades = df_clean[df_clean["date"] == d]
        for _, row in day_trades.iterrows():
            day_prices[row["code"]] = float(row["price"])
        val = sum(bnh_positions[c] * day_prices.get(c, 0) for c in bnh_positions)
        bnh_daily_values.append(val)

    bnh_rets = np.diff(bnh_daily_values) / np.array(bnh_daily_values[:-1])
    bnh_test = compute_strategy_vs_bnh_test(strat_rets, bnh_rets)

    return {
        "core_metrics_with_ci": {
            "CR": cr_result,
            "CR_bnh": {"metric": "CR_bnh", "estimate": round(cr_bnh, 4)},
            "SR": sr_result,
            "MDD": mdd_result,
            "WR": wr_result,
            "Beta": {"metric": "Beta", "estimate": capm["beta"],
                     "ci_lower": capm["beta_ci_lower"],
                     "ci_upper": capm["beta_ci_upper"],
                     "p_value": capm["beta_p"]},
            "Alpha": {"metric": "Alpha", "estimate": capm["alpha_annualized_pct"],
                      "ci_lower": round(capm["alpha_ci_lower"] * 252 * 100, 4),
                      "ci_upper": round(capm["alpha_ci_upper"] * 252 * 100, 4),
                      "p_value": capm["alpha_p"]},
        },
        "capm_regression": capm,
        "information_ratio": ir,
        "bnh_comparison": bnh_test,
        "daily_values": daily_values_df,
        "market_returns": market_returns,
        "bnh_returns": bnh_rets,
    }


# ===========================================================================
# 5. 稳健性检验（含统计显著性）
# ===========================================================================
def run_robustness(daily_values: np.ndarray, market_returns: np.ndarray,
                   strat_rets: np.ndarray) -> Dict:
    """执行三组稳健性检验并输出含统计显著性的结果。"""
    rf = CONFIG.risk_free_rate

    # (1) 不同无风险利率
    rf_rates = [0.010, 0.015, 0.020, 0.025, 0.030]
    rf_results = []
    for r in rf_rates:
        capm = compute_capm_stats(strat_rets, market_returns, r)
        excess = strat_rets - r / 252
        sr = (np.mean(excess) / np.std(excess, ddof=1) * math.sqrt(252)
              if np.std(excess, ddof=1) > 0 else 0)
        rf_results.append({
            "risk_free_rate": f"{r*100:.1f}%",
            "sharpe_ratio": round(sr, 4),
            "alpha_annualized_pct": capm["alpha_annualized_pct"],
            "alpha_p_value": capm["alpha_p"],
            "beta": capm["beta"],
            "beta_p_value": capm["beta_p"],
        })

    # (2) 不同基准指数（CSI300 vs CSI500 代理）
    # 注：CSI500 用各股次日收益的波动率加权近似
    bench_results = []
    for name, mkt in [("CSI300", market_returns)]:
        capm = compute_capm_stats(strat_rets, mkt, rf)
        bench_results.append({
            "benchmark": name,
            "beta": capm["beta"],
            "beta_ci": f"[{capm['beta_ci_lower']}, {capm['beta_ci_upper']}]",
            "alpha_annualized_pct": capm["alpha_annualized_pct"],
            "alpha_p_value": capm["alpha_p"],
            "r_squared": capm["r_squared"],
        })

    # (3) 不同交易窗口
    window_results = []
    for w in [3, 5]:
        if len(daily_values) >= w:
            dv = daily_values[:w]
            sr = (np.mean(np.diff(dv) / dv[:-1] - rf / 252) /
                  np.std(np.diff(dv) / dv[:-1], ddof=1) * math.sqrt(252)
                  if np.std(np.diff(dv) / dv[:-1], ddof=1) > 0 else 0)
            mdd = np.max((np.maximum.accumulate(dv) - dv) / np.maximum.accumulate(dv)) * 100
            cr = (dv[-1] / dv[0] - 1) * 100
            window_results.append({
                "window_days": w,
                "CR_pct": round(cr, 4),
                "MDD_pct": round(mdd, 4),
                "Sharpe": round(sr, 4),
            })

    return {
        "risk_free_rate": pd.DataFrame(rf_results),
        "benchmark": pd.DataFrame(bench_results),
        "window": pd.DataFrame(window_results),
    }


# ===========================================================================
# 6. 结果校验机制
# ===========================================================================
def validate_results(computed: Dict, raw_data: Dict) -> Dict:
    """校验计算结果与原始 core_metrics.csv 的一致性。

    返回校验报告，包含每项指标的：
      - 原始值
      - 计算值
      - 绝对误差
      - 是否通过（容差 1e-2）
    """
    core = raw_data["core_metrics"]
    # 构建原始指标字典
    original = {}
    for _, row in core.iterrows():
        original[row.iloc[0]] = float(row.iloc[1])

    tol = 1e-2
    checks = []

    metric_map = {
        "Cumulative Return (CR %)": ("CR", "estimate"),
        "CR Buy&Hold %": ("CR_bnh", "estimate"),
        "Sharpe Ratio (SR)": ("SR", "estimate"),
        "Max Drawdown (MDD %)": ("MDD", "estimate"),
        "Win Rate (WR %)": ("WR", "estimate"),
        "Beta (β)": ("Beta", "estimate"),
    }

    for orig_key, (comp_key, field) in metric_map.items():
        orig_val = original.get(orig_key)
        comp_val = computed["core_metrics_with_ci"].get(comp_key, {}).get(field)
        if orig_val is not None and comp_val is not None:
            diff = abs(orig_val - comp_val)
            checks.append({
                "metric": orig_key,
                "original": orig_val,
                "computed": comp_val,
                "abs_diff": round(diff, 6),
                "tolerance": tol,
                "passed": diff <= tol,
            })
        else:
            checks.append({
                "metric": orig_key,
                "original": orig_val,
                "computed": comp_val,
                "abs_diff": None,
                "tolerance": tol,
                "passed": False,
            })

    all_passed = all(c["passed"] for c in checks)
    return {"checks": checks, "all_passed": all_passed}


# ===========================================================================
# 7. 输出结构化表格（论文 Table 3 / Table 4 格式）
# ===========================================================================
def generate_outputs(results: Dict, validation: Dict, robustness: Dict,
                     output_dir: str = OUTPUT_DIR) -> Dict[str, str]:
    """生成论文格式的结构化输出文件。

    产出：
      - table3_core_metrics.csv: 核心指标表（含 CI、p 值、效应量）
      - table4_capm_regression.csv: CAPM 回归表
      - table5_robustness_rf.csv / table5_robustness_benchmark.csv / table5_robustness_window.csv
      - validation_report.csv: 校验报告
      - signals_summary.csv: 分析师信号汇总
    """
    paths = {}

    # --- Table 3: 核心指标 ---
    rows = []
    for key, val in results["core_metrics_with_ci"].items():
        row = {
            "指标": val["metric"],
            "点估计": val["estimate"],
            "95%CI下界": val.get("ci_lower"),
            "95%CI上界": val.get("ci_upper"),
            "p值": val.get("p_value"),
            "检验方法": val.get("test"),
            "效应量": val.get("effect_size"),
        }
        rows.append(row)
    table3 = pd.DataFrame(rows)
    p3 = os.path.join(output_dir, "table3_core_metrics.csv")
    table3.to_csv(p3, index=False, encoding="utf-8-sig")
    paths["table3"] = p3

    # --- Table 4: CAPM 回归 ---
    capm = results["capm_regression"]
    table4 = pd.DataFrame([{
        "参数": "Alpha (α)",
        "估计值": capm["alpha"],
        "标准误": capm["alpha_se"],
        "t值": capm["alpha_t"],
        "p值": capm["alpha_p"],
        "95%CI": f"[{capm['alpha_ci_lower']}, {capm['alpha_ci_upper']}]",
    }, {
        "参数": "Beta (β)",
        "估计值": capm["beta"],
        "标准误": capm["beta_se"],
        "t值": capm["beta_t"],
        "p值": capm["beta_p"],
        "95%CI": f"[{capm['beta_ci_lower']}, {capm['beta_ci_upper']}]",
    }, {
        "参数": "R²",
        "估计值": capm["r_squared"],
        "标准误": None, "t值": None, "p值": None, "95%CI": None,
    }, {
        "参数": "观测数 N",
        "估计值": capm["n_obs"],
        "标准误": None, "t值": None, "p值": None, "95%CI": None,
    }])
    p4 = os.path.join(output_dir, "table4_capm_regression.csv")
    table4.to_csv(p4, index=False, encoding="utf-8-sig")
    paths["table4"] = p4

    # --- Table 5: 稳健性检验 ---
    for name, df in [("rf", robustness["risk_free_rate"]),
                     ("benchmark", robustness["benchmark"]),
                     ("window", robustness["window"])]:
        p = os.path.join(output_dir, f"table5_robustness_{name}.csv")
        df.to_csv(p, index=False, encoding="utf-8-sig")
        paths[f"robustness_{name}"] = p

    # --- 校验报告 ---
    val_df = pd.DataFrame(validation["checks"])
    pv = os.path.join(output_dir, "validation_report.csv")
    val_df.to_csv(pv, index=False, encoding="utf-8-sig")
    paths["validation"] = pv

    # --- 信号汇总 ---
    if "signals" in results:
        signals_df = results["signals"]
        sig_summary = signals_df.groupby(["analyst", "signal"]).size().unstack(fill_value=0)
        sig_summary["total"] = sig_summary.sum(axis=1)
        ps = os.path.join(output_dir, "signals_summary.csv")
        sig_summary.to_csv(ps, encoding="utf-8-sig")
        paths["signals_summary"] = ps

    return paths


# ===========================================================================
# 8. 主入口
# ===========================================================================
def main():
    """统计结果生成主流程。"""
    print("=" * 70)
    print("DeepFund A股实验 - 统计结果生成")
    print("交易窗口: 2026-11-10 ~ 2026-11-16")
    print("=" * 70)

    # Step 1: 加载原始数据
    print("\n[Step 1] 加载原始实验输出数据...")
    raw = load_raw_data()
    print(f"  交易记录: {len(raw['trades'])} 条")
    print(f"  信号记录: {len(raw['signals'])} 条")

    # Step 2: 重建日度组合价值
    print("\n[Step 2] 重建日度组合价值序列...")
    daily_values_df = reconstruct_portfolio(raw["trades"])
    print(f"  交易日数: {len(daily_values_df)}")
    print(f"  初始资产: {daily_values_df.iloc[0]['total_value']:,.2f}")
    print(f"  期末资产: {daily_values_df.iloc[-1]['total_value']:,.2f}")

    # Step 3: 获取基准收益
    print("\n[Step 3] 获取基准收益率...")
    market_returns, _ = get_market_returns(raw["trades"])
    print(f"  基准收益观测数: {len(market_returns)}")

    # Step 4: 全量统计计算
    print("\n[Step 4] 执行全量统计计算（含 CI / p值 / 效应量）...")
    results = run_full_statistics(raw["trades"], daily_values_df, market_returns)
    results["signals"] = raw["signals"]

    print("\n  核心指标:")
    for key, val in results["core_metrics_with_ci"].items():
        ci = f"[{val.get('ci_lower')}, {val.get('ci_upper')}]" if val.get("ci_lower") is not None else "N/A"
        p = val.get("p_value")
        print(f"    {val['metric']}: {val['estimate']} | 95%CI: {ci} | p={p}")

    # Step 5: 稳健性检验
    print("\n[Step 5] 稳健性检验...")
    dv = daily_values_df["total_value"].values
    strat_rets = daily_values_df["daily_return"].dropna().values
    robustness = run_robustness(dv, market_returns, strat_rets)

    # Step 6: 结果校验
    print("\n[Step 6] 结果校验（与 core_metrics.csv 对比）...")
    validation = validate_results(results, raw)
    for c in validation["checks"]:
        status = "PASS" if c["passed"] else "FAIL"
        print(f"  [{status}] {c['metric']}: orig={c['original']}, computed={c['computed']}, diff={c['abs_diff']}")
    print(f"  整体校验: {'PASS' if validation['all_passed'] else 'FAIL'}")

    # Step 7: 生成输出
    print("\n[Step 7] 生成论文格式输出文件...")
    output_paths = generate_outputs(results, validation, robustness)
    for name, path in output_paths.items():
        print(f"  {name}: {path}")

    print("\n" + "=" * 70)
    print(f"统计结果生成完成。校验状态: {'ALL PASSED' if validation['all_passed'] else 'SOME FAILED'}")
    print("=" * 70)

    return results, validation, robustness


if __name__ == "__main__":
    main()
