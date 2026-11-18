"""
实验配置模块
============
根据论文《Time Travel is Cheating: Going Live with DeepFund for Real-Time Fund
Investment Benchmarking》的实验设定，在A股市场环境下复现实验。

# 本模块定义：
1. 5只符合论文筛选标准（巴菲特Q1 2025前五大持仓逻辑）的A股标的
2. 论文Table 6中的全部参数（交易窗口由24天改为5天）
3. A股市场对应的无风险利率、基准指数等市场参数
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List


# ---------------------------------------------------------------------------
# 1. 样本股票筛选（对应论文：Berkshire Hathaway Q1 2025 前五大持仓）
# ---------------------------------------------------------------------------
# 论文选取 Buffett 前五大持仓，覆盖科技(AAPL)、金融服务(AXP)、银行(BAC)、
# 必需消费品(KO)、能源(CVX) 五个行业的大盘蓝筹。
# A股对标：选取五个不同行业的代表性大盘蓝筹股，确保行业分散度与论文一致。
# ---------------------------------------------------------------------------
SAMPLE_STOCKS: List[Dict] = [
    {
        "code": "600519",
        "name": "贵州茅台",
        "sector": "Consumer Staples / 白酒",
        "buffett_counterpart": "KO (Coca-Cola)",
        "rationale": "A股必需消费品龙头，对应论文中可口可乐(KO)的消费防御属性",
    },
    {
        "code": "600036",
        "name": "招商银行",
        "sector": "Banking / 银行",
        "buffett_counterpart": "BAC (Bank of America)",
        "rationale": "A股股份制银行龙头，对应论文中美国银行(BAC)的金融板块配置",
    },
    {
        "code": "601318",
        "name": "中国平安",
        "sector": "Financial Services / 保险",
        "buffett_counterpart": "AXP (American Express)",
        "rationale": "A股综合金融龙头，对应论文中美国运通(AXP)的金融服务属性",
    },
    {
        "code": "300750",
        "name": "宁德时代",
        "sector": "Technology / 新能源",
        "buffett_counterpart": "AAPL (Apple)",
        "rationale": "A股新能源科技龙头，对应论文中苹果(AAPL)的科技成长属性",
    },
    {
        "code": "600028",
        "name": "中国石化",
        "sector": "Energy / 石油石化",
        "buffett_counterpart": "CVX (Chevron)",
        "rationale": "A股能源板块龙头，对应论文中雪佛龙(CVX)的能源板块配置",
    },
]


# ---------------------------------------------------------------------------
# 2. 实验参数（论文 Table 6: DeepFund Parameter Settings）
# ---------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    # ---- 投资组合配置（论文 Section 3）----
    initial_cash: float = 1_000_000.0       # 初始资金 100万人民币（论文为$100,000）
    num_stocks: int = 5                     # 持仓股票数量

    # ---- 交易周期（论文 Section 3，24天 → 改为5天）----
    trading_days: int = 5                   # 交易天数（论文为24天，按要求改为5天）
    # 交易起止日期：取最近5个交易日（由数据模块动态确定）

    # ---- 论文 Table 6 参数 ----
    llm_temperature: float = 0.5            # LLM 温度参数
    retry_times: int = 3                    # 重试次数
    technical_window: int = 100             # 技术分析OHLCV覆盖天数
    insider_count: int = 10                 # 内部人交易分析条数
    num_news: int = 10                      # 新闻分析条数
    decision_memory_size: int = 5           # 决策记忆大小

    # ---- 市场参数（A股适配）----
    risk_free_rate: float = 0.015            # 无风险利率（中国1年期国债收益率约1.5%）
    benchmark_index: str = "000300"          # 基准指数：沪深300（对标论文S&P 500）
    benchmark_name: str = "CSI 300"

    # ---- 交易成本（论文未计入，A股实验按实际设置）----
    commission_rate: float = 0.0003          # 佣金费率万分之三
    stamp_tax_rate: float = 0.0005           # 印花税千分之零点五（卖出单边）

    # ---- 风控参数（仓位管理）----
    max_single_position: float = 0.30        # 单只股票最大仓位比例 30%
    min_cash_ratio: float = 0.10             # 最低现金保留比例 10%


CONFIG = ExperimentConfig()
