# DeepFund-AStock · A 股多智能体投研与实盘交易基准

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-1C3C3C)](https://github.com/langchain-ai/langgraph)
[![Data](<https://img.shields.io/badge/Data-AKShare%20%2B%20Eastmoney-E4392E>)](https://akshare.akfamily.xyz/)
[![DB](<https://img.shields.io/badge/Storage-Supabase%20%7C%20SQLite-3ECF8E>)](https://supabase.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

---

## 1. 项目简介

本项目是[HKUSTDial/DeepFund](https://github.com/HKUSTDial/DeepFund) 面向**A 股市场复现与实践**，核心思想是：让一个由多个 LLM Agent 组成的"虚拟基金"按照**真实时间顺序**（而非一次性喂入全部历史数据）逐个交易日做决策，从而杜绝"时间旅行"式的前视偏差（look-ahead bias），并用统一指标衡量其真实投资能力。

> **本项目全程采用Trae编写+人工校核，并在持续测试、核验与优化。**

本仓库在保留原论文方法论的前提下，把数据层、分析师角色、交易与存储全部落到中国 A 股市场：

- **数据层**：从美股（Alpha Vantage / Yahoo Finance）切换为 A 股（AKShare + 东方财富直连）
- **分析师**：6 位 LLM 分析师覆盖技术面、基本面、公司新闻、宏观、政策、内部人交易
- **决策层**：投资组合经理（Portfolio Manager）先做风险控制再出交易指令，带 5 期决策记忆
- **存储层**：Supabase（云端 Postgres）或本地 SQLite，完整留存 config / portfolio / decision / signal 四类记录
- **研究层**：内置实验复现、统计推断（Bootstrap 置信区间、p 值、效应量）、CAPM / Fama-French 回归与稳健性检验

### 主要能力

| 能力              | 说明                                                                              |
| ----------------- | --------------------------------------------------------------------------------- |
| 逐日实盘式回测    | 每个交易日只使用"当日及之前"的数据，`trading_date` 贯穿所有数据接口             |
| 多 Agent 协同研判 | 6 位分析师独立出信号（Bullish / Bearish / Neutral），Portfolio Manager 聚合后决策 |
| 可解释决策留存    | 每条信号与决策都保存完整 `justification` 与 `llm_prompt`，可追溯              |
| 组合级风控        | 单标的仓位上限按标的数量动态计算，超出区间自动裁剪                                |
| 双数据库支持      | Supabase 云端持久化 / SQLite 本地离线                                             |
| 学术级统计产出    | 论文 Appendix D 全部 7 项指标 + 置信区间 + 显著性检验 + 稳健性表格                |

---

## 2. 系统架构

### 2.1 决策工作流（LangGraph）

```
                    ┌──────────────────────────┐
                    │      FundState (state)   │
                    │  exp_name / trading_date │
                    │  ticker / portfolio / ...│
                    └────────────┬─────────────┘
                                 │  START
        ┌──────────┬──────────┬──┴───────┬──────────┬──────────┐
        ▼          ▼          ▼          ▼          ▼          ▼
   technical  fundamental  company_news  policy  macroeconomic  insider
        │          │          │          │          │          │
        └──────────┴──────────┴────┬─────┴──────────┴──────────┘
                                  ▼
                    ┌──────────────────────────┐
                    │   portfolio manager      │
                    │  1) 风险控制 → 目标仓位   │
                    │  2) 决策记忆 → 交易指令   │
                    └────────────┬─────────────┘
                                 ▼  END
                           Decision(Buy/Sell/Hold)
```

- 图定义见 [workflow.py](file:///src/graph/workflow.py)，节点通过 [registry.py](file:///d:/02Github/06MyTrade/DeepFund-AStock/src/agents/registry.py) 注册。
- 每个 ticker 独立跑一遍完整图，结果写回组合后再处理下一个 ticker（**顺序执行**，因此耗时随标的数线性增长）。
- 可选 `planner_mode`：由 planner agent 从已验证分析师列表中挑选本次要跑的角色（见 [planner.yaml](file:///d:/02Github/06MyTrade/DeepFund-AStock/src/config/planner.yaml)）。

### 2.2 数据流

```
AKShare / 东财直连  ──►  apis/akshare/api.py  ──►  Router  ──►  Analyst Agent
                                                                    │
                                                              AnalystSignal
                                                                    ▼
                                                        Portfolio Manager
                                                                    │
                                                             Decision
                                                                    ▼
                                              Database (Supabase / SQLite)
                                                                    │
                                                                    ▼
                                              experiment/ 统计与可视化
```

---

## 3. 技术栈

### 3.1 运行时与编排

| 类别       | 组件                | 版本 / 说明                                                            |
| ---------- | ------------------- | ---------------------------------------------------------------------- |
| 语言       | Python              | `>=3.11`（`environment.yml` 锁定 3.11.11）                         |
| 工作流编排 | **LangGraph** | `0.3.1`，`StateGraph` + `TypedDict` 状态                         |
| 状态建模   | **Pydantic**  | `2.10.6`，`Decision` / `AnalystSignal` / `Portfolio` 等 schema |
| 类型扩展   | typing-extensions   | `4.12.2`                                                             |

### 3.2 LLM 接入

| 组件                    | 说明                                                                       |
| ----------------------- | -------------------------------------------------------------------------- |
| `langchain-openai`    | OpenAI 及所有 OpenAI 兼容端点（Qwen / 智谱 / MiniMax / YiZhan / AiHubMix） |
| `langchain-anthropic` | Anthropic Claude                                                           |
| `langchain-deepseek`  | DeepSeek 官方                                                              |
| `langchain-ollama`    | 本地 Ollama                                                                |
| `langchain-fireworks` | Fireworks AI                                                               |

支持 10 家 provider，映射逻辑见 [provider.py](file:///d:/02Github/06MyTrade/DeepFund-AStock/src/llm/provider.py)：

`OpenAI` · `Anthropic` · `DeepSeek` · `Alibaba(Qwen)` · `ZhiPu` · `MiniMax` · `Ollama` · `Fireworks` · `YiZhan` · `AiHubMix`

### 3.3 数据与存储

| 类别     | 组件                                            | 说明                                  |
| -------- | ----------------------------------------------- | ------------------------------------- |
| A 股数据 | **AKShare**                               | 日 K、估值、新闻、宏观                |
| 东财直连 | `requests` + `datacenter-web.eastmoney.com` | 董监高持股变动（分页 JSON，可控窗口） |
| 数据处理 | pandas `2.2.3` / numpy `2.2.3`              | 时序与矩阵运算                        |
| 云端存储 | **Supabase** `2.15.0`                   | Postgres + JSONB                      |
| 本地存储 | **SQLite**（标准库 `sqlite3`）          | 零依赖离线模式                        |
| 配置     | PyYAML `6.0.2` + python-dotenv `1.0.1`      | YAML 实验配置 + `.env` 密钥         |

### 3.4 研究 / 统计（可选依赖）

统计与绘图模块使用以下库，**未包含在 `pyproject.toml` 主依赖中**，需按需安装：

| 组件            | 用途                                |
| --------------- | ----------------------------------- |
| `statsmodels` | CAPM / Fama-French OLS 回归、t 检验 |
| `scipy`       | 分布、二项检验、置信区间            |
| `matplotlib`  | 图表输出                            |
| `pytest`      | 单元测试                            |

```bash
uv pip install statsmodels scipy matplotlib pytest
```

---

## 4. 目录结构

```
DeepFund-AStock/
├── src/
│   ├── main.py                    # 唯一入口：--config / --trading-date / --local-db
│   ├── agents/
│   │   ├── analysts/
│   │   │   ├── technical.py       # 技术面（趋势/均值回归/RSI/波动率/量能/支撑阻力）
│   │   │   ├── fundamental.py     # 基本面（PE/PB/PS/PCF/ROE/增速）
│   │   │   ├── company_news.py    # 公司新闻
│   │   │   ├── policy.py          # 政策新闻
│   │   │   ├── macroeconomic.py   # 宏观经济指标
│   │   │   └── insider.py         # 董监高持股变动
│   │   ├── portfolio_manager.py   # 风险控制 + 交易决策
│   │   ├── planner.py             # 可选：动态挑选分析师
│   │   └── registry.py            # Agent 注册表
│   ├── apis/
│   │   ├── akshare/api.py         # A 股数据实现（主力）
│   │   ├── alphavantage/          # 美股（原版保留）
│   │   ├── yfinance/              # 美股（原版保留）
│   │   ├── financialdataset/      # 美股基本面（原版保留）
│   │   └── router.py              # 数据源路由
│   ├── graph/
│   │   ├── workflow.py            # LangGraph 拓扑与运行
│   │   ├── schema.py              # FundState / Decision / Position 等
│   │   └── constants.py           # AgentKey / Signal / Action 枚举
│   ├── llm/
│   │   ├── inference.py           # LLMConfig / get_model / agent_call
│   │   ├── provider.py            # provider → LangChain 客户端映射
│   │   └── prompt.py              # 全部 Prompt 模板
│   ├── database/
│   │   ├── interface.py           # BaseDB 抽象
│   │   ├── supabase_helper.py     # Supabase 实现
│   │   ├── sqlite_helper.py       # SQLite 实现
│   │   ├── sqlite_setup.py        # 本地库建表
│   │   └── supabase_setup.sql     # 云端建表 SQL
│   ├── util/
│   │   ├── config.py              # YAML 配置解析
│   │   ├── db_helper.py           # DB 全局单例
│   │   └── logger.py              # 结构化日志
│   ├── config/
│   │   ├── a_stock.yaml           # A 股示例（600519 / 000001）
│   │   ├── dev.yaml               # 美股示例（TSLA / NVDA）
│   │   ├── planner.yaml           # planner 模式示例
│   │   ├── exp/
│   │   │   ├── buffett.yaml       # 论文同款：AAPL/AXP/BAC/KO/CVX
│   │   │   └── gold_etf.yaml      # 黄金 ETF / 金矿股
│   │   └── provider/              # 各 LLM 供应商配置（11 个）
│   ├── experiment/                # 研究复现与统计模块（见第 9 节）
│   ├── logs/                      # 运行日志 deepfund_YYYYMMDD_HHMMSS.log
│   └── example/                   # 示例 SQLite 库与历史日志
├── pyproject.toml
├── environment.yml
├── uv.lock
├── TECHNICAL_GUIDE.md             # 开发者技术说明
└── LICENSE                        # MIT
```

---

## 5. 环境要求

| 项目     | 要求                                                                                  |
| -------- | ------------------------------------------------------------------------------------- |
| 操作系统 | Windows / macOS / Linux（Windows 已在 PowerShell 5 验证）                             |
| Python   | **3.11 及以上**（推荐 3.11.11，与 `environment.yml` 一致）                    |
| 网络     | 能访问 `akshare` 数据源与东财 `datacenter-web.eastmoney.com`；能访问所用 LLM 端点 |
| 数据库   | Supabase 项目（默认）或本地 SQLite（`--local-db`）                                  |
| 磁盘     | ≥ 1 GB（依赖 + 日志 + 输出）                                                         |

---

## 6. 安装部署

### 步骤 1：获取代码

```bash
git clone <your-repo-url> DeepFund-AStock
cd DeepFund-AStock
```

### 步骤 2：创建环境（三选一）

**方式 A：uv（推荐，仓库自带 `uv.lock`）**

```bash
uv sync
```

**方式 B：Conda**

```bash
conda env create -f environment.yml
conda activate deepfund
```

**方式 C：已有虚拟环境，仅装依赖**

```bash
uv venv
uv pip install -r pyproject.toml
# 研究/统计模块另需
uv pip install statsmodels scipy matplotlib pytest
```

### 步骤 3：配置 `.env`

复制模板并填写：

```bash
cp .env.example .env
```

`.env` 关键项：

```ini
# ── 数据库 ─────────────────────────────────
# 云端（默认）：需先在 Supabase 执行 database/supabase_setup.sql
SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_KEY=your-supabase-key

# 本地 SQLite（使用 --local-db 时生效）
DB_PATH=assets/deepfund.db

# ── LLM（按所选 provider 填其一）─────────────
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
DEEPSEEK_API_KEY=...
QWEN_API_KEY=...
ZHIPU_API_KEY=...
MINIMAX_API_KEY=...
# MiniMax 端点多套域名，按账号归属选择：
#   国内平台 https://api.minimaxi.com/v1
#   海外平台 https://api.minimax.io/v1
#   默认     https://api.minimax.cn/v1
MINIMAX_BASE_URL=https://api.minimaxi.com/v1

# 第三方网关
YIZHAN_API_KEY=...
AIHUBMIX_API_KEY=...
FIREWORKS_API_KEY=...

# ── 美股备用数据源（仅美股实验需要）───────────
ALPHA_VANTAGE_API_KEY=...
```

### 步骤 4：初始化数据库

**Supabase（默认）**：在 Supabase 控制台 SQL Editor 执行 [supabase_setup.sql](file:///d:/02Github/06MyTrade/DeepFund-AStock/src/database/supabase_setup.sql)，创建 `config` / `portfolio` / `decision` / `signal` 四张表及索引。

**SQLite（本地）**：

```bash
cd src
python database/sqlite_setup.py
```

### 步骤 5：验证安装

```bash
cd src
python -c "from graph.workflow import AgentWorkflow; print('ok')"
```

---

## 7. 配置说明

### 7.1 实验配置（YAML）

配置位于 `src/config/`，由 [config.py](file:///d:/02Github/06MyTrade/DeepFund-AStock/src/util/config.py) 解析（显式 UTF-8，兼容中文注释）。

```yaml
exp_name: "a_stock"          # 实验名，DB 中作为 config 的唯一键
cashflow: 100000             # 初始资金
tickers:                     # 标的列表（A 股 6 位代码 / 美股代码）
  - "600519"
  - "000001"
workflow_analysts:           # 本次要跑的分析师，可选：
  - fundamental              #   technical / fundamental / company_news
  - macroeconomic            #   policy / macroeconomic / insider
  - technical
  - company_news
  - insider
  - policy
planner_mode: false          # 可选：true 时由 planner 动态挑选分析师
llm:
  provider: "DeepSeek"       # 见 3.2 provider 列表（大小写敏感）
  model: "deepseek-chat"
  temperature: 0.5           # 可选，默认 0.5；各 provider 有合法区间校验
  max_tokens: 6000           # 可选；推理模型需留足 think 预算
```

仓库内置配置：

| 文件                                                                                       | 标的                        | 说明                                                   |
| ------------------------------------------------------------------------------------------ | --------------------------- | ------------------------------------------------------ |
| [a_stock.yaml](file:///d:/02Github/06MyTrade/DeepFund-AStock/src/config/a_stock.yaml)       | 600519 / 000001             | A 股最小示例                                           |
| [dev.yaml](file:///d:/02Github/06MyTrade/DeepFund-AStock/src/config/dev.yaml)               | TSLA / NVDA                 | 美股开发调试                                           |
| [buffett.yaml](file:///d:/02Github/06MyTrade/DeepFund-AStock/src/config/exp/buffett.yaml)   | AAPL / AXP / BAC / KO / CVX | 论文同款 5 只                                          |
| [gold_etf.yaml](file:///d:/02Github/06MyTrade/DeepFund-AStock/src/config/exp/gold_etf.yaml) | GLD / NEM / GOLD            | 黄金 ETF 与金矿股                                      |
| [planner.yaml](file:///d:/02Github/06MyTrade/DeepFund-AStock/src/config/planner.yaml)       | TSLA / NVDA                 | 演示 planner 模式                                      |
| `config/provider/*.yaml`                                                                 | —                          | 10 份 provider 专用配置（含 temperature / max_tokens） |

### 7.2 环境变量

| 变量                                | 必填                | 说明                             |
| ----------------------------------- | ------------------- | -------------------------------- |
| `SUPABASE_URL` / `SUPABASE_KEY` | 默认模式必填        | 云端数据库                       |
| `DB_PATH`                         | `--local-db` 必填 | SQLite 文件路径                  |
| `<PROVIDER>_API_KEY`              | 是                  | 与 YAML 中 `llm.provider` 匹配 |
| `MINIMAX_BASE_URL`                | 否                  | 覆盖 MiniMax 默认端点            |

---

## 8. 使用说明

### 8.1 运行一次交易决策

> 必须在 `src/` 目录下执行（代码使用 `graph.*` / `util.*` 顶层包导入）。

```bash
cd src

# Supabase（默认）
python main.py --config ./config/provider/minimax.yaml --trading-date 2026-11-16

# 本地 SQLite
python main.py --config ./config/a_stock.yaml --trading-date 2026-11-16 --local-db
```

参数：

| 参数               | 必填 | 说明                        |
| ------------------ | ---- | --------------------------- |
| `--config`       | 是   | YAML 配置路径               |
| `--trading-date` | 是   | 交易日，格式 `YYYY-MM-DD` |
| `--local-db`     | 否   | 使用 SQLite 而非 Supabase   |

### 8.2 运行机制（重要）

1. **组合继承**：读取该 `exp_name` 下最近一条 portfolio，复制为新组合（新 `portfolio_id`），保证历史可追溯。
2. **时序校验**：若 `--trading-date` 早于库中最新交易日，程序直接报错，防止时间倒流。
3. **逐标的决策**：每个 ticker 依次执行 `6 位分析师 → 风险控制 → 交易决策`，每只约 8 次 LLM 调用。
4. **组合落库**：全部 ticker 处理完后，写回 `portfolio`（`cashflow` / `total_assets` / `positions`）。
5. **日志**：`src/logs/deepfund_YYYYMMDD_HHMMSS.log`，含每位分析师信号、风控结论、最终决策与耗时。

### 8.3 分析师一览

| 分析师            | 关注点                                      | 数据来源                            | 支持历史日期    |
| ----------------- | ------------------------------------------- | ----------------------------------- | --------------- |
| `technical`     | 趋势、均值回归、RSI、波动率、量能、支撑阻力 | 个股日 K（前复权）                  | ✅              |
| `fundamental`   | PE / PB / PS / PCF / ROE / 增速             | 个股信息 + 估值快照                 | ⚠️ 仅最新快照 |
| `company_news`  | 公司公告与媒体报道                          | AKShare 个股新闻                    | ✅              |
| `policy`        | 宏观与产业政策                              | 全球财经快讯 + 新闻联播             | ✅              |
| `macroeconomic` | GDP / CPI / PMI / M2 / LPR                  | AKShare 宏观接口                    | ⚠️ 最新已发布 |
| `insider`       | 董监高持股变动（最近 N 笔）                 | 东财 `RPT_EXECUTIVE_HOLD_DETAILS` | ✅              |

所有分析师统一输出 `Signal = Bullish / Bearish / Neutral` + `justification`。

### 8.4 投资组合经理与风控

见 [portfolio_manager.py](file:///d:/02Github/06MyTrade/DeepFund-AStock/src/agents/portfolio_manager.py)：

- **单标的上限**：`max_position_ratio = round(2 / N × 20) / 20`（N 为标的数，即"自身基准仓位 + 一个其他标的的份额"，取 0.05 的整数倍）。N=5 时为 0.40；N=1 时为 1.0。
- **风险控制**：LLM 给出 `optimal_position_ratio`，越界自动裁剪到 `[0, max_position_ratio]`。
- **可交易股数**：`min(目标市值缺口, 现金) // 当前价`（买入）/ `max(目标市值缺口 // 当前价, -当前持仓)`（卖出）。
- **决策记忆**：取该 ticker 最近 **5** 条决策作为上下文。
- **成交假设**：按当日收盘价成交，**不计佣金与印花税**；股数为整数股（不做 100 股整手约束）。

### 8.5 数据库表结构

| 表            | 关键字段                                                                                                             | 说明                                               |
| ------------- | -------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| `config`    | `exp_name`, `tickers`, `has_planner`, `llm_model`, `llm_provider`                                          | 实验配置（按 `exp_name` 唯一）                   |
| `portfolio` | `id`, `config_id`, `trading_date`, `cashflow`, `total_assets`, `positions(JSONB)`                        | 每交易日一条组合快照                               |
| `decision`  | `portfolio_id`, `trading_date`, `ticker`, `action`, `shares`, `price`, `llm_prompt`, `justification` | 交易决策                                           |
| `signal`    | `portfolio_id`, `ticker`, `analyst`, `signal`, `llm_prompt`, `justification`                             | 分析师信号（无独立日期，经 `portfolio_id` 关联） |

---

## 9. 实验复现与统计分析模块

`src/experiment/` 提供统计原语与 Supabase 统计脚本：

| 脚本                                                                                               | 数据源                               | 输出                 | 用途                                                                              |
| -------------------------------------------------------------------------------------------------- | ------------------------------------ | -------------------- | --------------------------------------------------------------------------------- |
| [statistics.py](file:///d:/02Github/06MyTrade/DeepFund-AStock/src/experiment/statistics.py)         | —                                   | —                   | 统计原语库：CR/SR/MDD/WR/Beta/Alpha、Bootstrap 置信区间、CAPM、效应量、稳健性检验 |
| [config.py](file:///d:/02Github/06MyTrade/DeepFund-AStock/src/experiment/config.py)                 | —                                   | —                   | 标的池与统计参数（`CONFIG` / `SAMPLE_STOCKS`）                                |
| [supabase_stats.py](file:///d:/02Github/06MyTrade/DeepFund-AStock/src/experiment/supabase_stats.py) | **Supabase 真实 LLM 运行记录** | `output_supabase/` | 对真实实验结果做完整统计分析                                                      |

### 9.1 指标与统计方法（对应论文 Appendix D）

| 指标       | 定义                                           | 统计推断                           |
| ---------- | ---------------------------------------------- | ---------------------------------- |
| `CR`     | 累计收益率 `(V_T / V_0 − 1) × 100%`        | Bootstrap 95% CI + 单样本 t 检验   |
| `CR_bnh` | 首日等权买入持有收益率                         | —                                 |
| `SR`     | 夏普比率 `mean(r−r_f)/std(r−r_f) × √252` | Bootstrap CI + t 检验 + Cohen's d  |
| `MDD`    | 最大回撤                                       | Bootstrap CI                       |
| `WR`     | 胜率（Buy 后次日涨 / Sell 后次日跌）           | 二项检验 + Clopper-Pearson 精确 CI |
| `β`     | CAPM 斜率                                      | OLS t 检验 + 95% CI                |
| `α`     | CAPM 截距（年化）                              | OLS t 检验 + 95% CI                |

补充：信息比率 `IR`、跟踪误差、策略 vs 买入持有的配对 t 检验；CAPM 与 Fama-French 三因子回归；无风险利率 / 基准指数 / 交易窗口三组稳健性检验。Bootstrap 固定 `seed=42`，重采样 10,000 次，保证可复现。

### 9.2 基于 Supabase 的运行方式

```bash
cd src/experiment
python supabase_stats.py \
  --exp-name minimax-test \
  --start 2026-11-10 \
  --end 2026-11-16 \
  --output-dir ./output_supabase
```

内置三层数据校验，全部通过才视为结果可信：

| 校验             | 内容                                             |
| ---------------- | ------------------------------------------------ |
| A 资产恒等式     | `total_assets == cashflow + Σ position.value` |
| B 持仓累计一致性 | 第 d 日 DB 持仓 == 前 d 日决策净额累计           |
| C 决策完整性     | 每交易日决策数 == 标的数，且标的集合匹配         |

> 同一交易日若存在多次运行产生的多条 portfolio，脚本自动保留 `updated_at` 最新的一条（即最终完成的那次），避免中断记录污染时间序列。

### 9.3 输出文件

`output_supabase/` 下同时落盘**原始记录**与**统计结果**：

- 原始：`portfolio_history.csv`、`positions_history.csv`、`price_matrix.csv`、`decisions.csv`、`signals.csv`
- 统计：`table3_core_metrics.csv`、`table4_capm_regression.csv`、`table5_robustness_*.csv`、`signals_summary.csv`、`validation_report.csv`

### 9.4 本期实验：设置与结果

以下为仓库内已完成的一次真实 LLM 实验（数据可由 `supabase_stats.py` 一键复现）。

#### 实验设置

| 项目              | 取值                                                                                        | 与原论文差异                                             |
| ----------------- | ------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| 实验名            | `minimax-test`                                                                            | —                                                       |
| LLM               | MiniMax / **MiniMax-M3**                                                              | 论文使用 GPT-4o 系列                                     |
| `temperature`   | **1.0**                                                                               | 论文 Table 6 为 **0.5**                            |
| `max_tokens`    | 6000（含 `<think>` 推理 token）                                                           | —                                                       |
| 初始资金          | ¥100,000                                                                                   | —                                                       |
| 标的              | 600519 贵州茅台 · 600036 招商银行 · 601318 中国平安 · 300750 宁德时代 · 600028 中国石化 | 论文为巴菲特 Q1 2025 前五大持仓                          |
| 分析师            | 6 位：technical / fundamental / company_news / policy / macroeconomic / insider             | 论文日常交易使用 4 位（不含 fundamental、macroeconomic） |
| 交易窗口          | 2026-11-10 ~ 2026-11-17（**6 个交易日**）                                             | 论文为 24 个交易日                                       |
| 基准 / 无风险利率 | 沪深300 / 1.5%                                                                              | 论文为 S&P 500 / 美债                                    |
| 日收益观测数      | **N = 5**                                                                             | 论文 N = 23                                              |
| 运行方式          | 每日执行一次 `src/main.py`，组合逐日继承                                                  | —                                                       |
| 单轮耗时          | 约 14 分钟（5 标的串行，含推理 token）                                                      | —                                                       |

#### 资产曲线

| 交易日     | 现金        | 总资产                 | 日收益    |
| ---------- | ----------- | ---------------------- | --------- |
| 2026-11-10 | ¥31,416.98 | ¥100,000.00           | —        |
| 2026-11-11 | ¥34,568.22 | ¥99,129.23            | −0.8708% |
| 2026-11-12 | ¥24,567.39 | ¥98,662.31            | −0.4710% |
| 2026-11-13 | ¥10,816.07 | ¥99,222.83            | +0.5681%  |
| 2026-11-16 | ¥16,706.21 | ¥100,413.80           | +1.2003%  |
| 2026-11-17 | ¥20,652.54 | **¥100,453.75** | +0.0398%  |

#### 交易行为

6 个交易日共 30 条决策：**Buy 16 / Sell 11 / Hold 3**。

| 交易日     | Buy | Sell | Hold |
| ---------- | --- | ---- | ---- |
| 2026-11-10 | 5   | 0    | 0    |
| 2026-11-11 | 2   | 2    | 1    |
| 2026-11-12 | 3   | 1    | 1    |
| 2026-11-13 | 3   | 2    | 0    |
| 2026-11-16 | 1   | 3    | 1    |
| 2026-11-17 | 2   | 3    | 0    |

#### 分析师信号分布（每类 30 条）

| 分析师        | Bullish | Bearish | Neutral |
| ------------- | ------- | ------- | ------- |
| fundamental   | 21      | 0       | 9       |
| policy        | 18      | 0       | 12      |
| company_news  | 10      | 2       | 18      |
| technical     | 6       | 6       | 18      |
| insider       | 2       | 1       | 27      |
| macroeconomic | 0       | 30      | 0       |

#### 核心指标（论文 Appendix D）

| 指标           | 点估计               | 95% CI                 | p 值   | 检验方法                   |
| -------------- | -------------------- | ---------------------- | ------ | -------------------------- |
| CR             | **0.4538%**    | [−2.6188, 3.7272]     | 0.8122 | 单样本 t 检验              |
| CR_bnh         | **0.4546%**    | —                     | —     | —                         |
| SR             | **1.6869**     | [−22.6083, 22.8664]   | 0.8239 | 单样本 t 检验（Cohen's d） |
| MDD            | **1.3377%**    | [0.0000, 2.6575]       | —     | Bootstrap                  |
| WR（已平仓）   | **40.9091%**   | [22.7273, 63.6364]     | 0.5235 | 二项检验                   |
| WR（含未平仓） | 33.3333%             | [14.8148, 51.8519]     | 0.1221 | 二项检验                   |
| β             | **−1.0348**   | [−2.9733, 0.9038]     | 0.1879 | OLS t 检验                 |
| α（年化）     | **−86.2877%** | [−402.8472, 230.2776] | 0.4495 | OLS t 检验                 |

#### CAPM 回归

| 参数        | 估计值     | 标准误          | t 值     | p 值   |
| ----------- | ---------- | --------------- | -------- | ------ |
| α（日）    | −0.003424 | 0.003947        | −0.8675 | 0.4495 |
| α（年化）  | −86.2877% | —              | —       | 0.4495 |
| β          | −1.0348   | 0.6091          | −1.6987 | 0.1879 |
| R²         | 0.4903     | —              | —       | —     |
| N           | 5          | —              | —       | —     |
| 信息比率 IR | 6.2598     | 跟踪误差 0.2023 | —       | —     |

#### 期末持仓（2026-11-17）

| 标的            | 持股  | 市值        | 权重   |
| --------------- | ----- | ----------- | ------ |
| 600036 招商银行 | 578   | ¥25,114.10 | 25.00% |
| 601318 中国平安 | 367   | ¥20,056.55 | 19.97% |
| 600519 贵州茅台 | 10    | ¥14,577.00 | 14.51% |
| 600028 中国石化 | 1,704 | ¥10,036.56 | 9.99%  |
| 300750 宁德时代 | 42    | ¥10,017.00 | 9.97%  |
| 现金            | —    | ¥20,652.54 | 20.56% |

#### 数据校验

| 校验             | 结果                 |
| ---------------- | -------------------- |
| A 资产恒等式     | 6 / 6 通过           |
| B 持仓累计一致性 | 30 / 30 通过         |
| C 决策完整性     | 6 / 6 通过           |
| 整体             | **ALL PASSED** |

#### 结果解读

- **CR 与 CR_bnh 基本持平**（0.4538% vs 0.4546%），即 6 个交易日内组合表现与"首日等权买入并持有"无实质差异，未产生超额收益。
- **α 为负且不显著**（年化 −86.29%，p = 0.4495），β 为 −1.03 且不显著（p = 0.1879），R² 仅 0.49；在 N=5 的样本下，任何方向性结论都不具备统计功效。
- **WR 40.91%** 低于 50% 但二项检验不显著（p = 0.5235），与"信号以 Bullish 为主、但短期涨跌随机"的分布一致。
- **组合结构**：期末现金 20.56%，招商银行达到 25% 的目标权重上限，中国平安接近 20%，呈现"低估值蓝筹 + 高现金"的防御性配置。
- **方法学提醒**：6 个交易日只有 5 个日收益观测，Bootstrap 置信区间极宽（如 SR 的 CI 跨越 ±22）。上述 p 值仅作论文格式展示，**不具备统计推断意义**，应作为描述性结果呈现。

---

## 10. 已知限制

- **样本量**：本期实验为 6 个交易日、仅 5 个日收益观测，任何显著性检验的功效都极低，p 值不具备推断意义，应作为描述性结果呈现。
- **成交假设**：按收盘价成交，不计佣金/印花税，不做 100 股整手与涨跌停约束。
- **分析师可复现性**：LLM 在 `temperature > 0` 下输出不完全确定，同一输入可能得到不同信号。
- **美股路径**：`yfinance` / `alphavantage` / `financialdataset` 为上游保留能力，A 股实验走 `akshare` 路由。
- **串行执行**：标的逐个处理，标的多时耗时长。

---

## 11. 待办事项（Roadmap & TODO）

### 11.1 A 股操作模块

- [ ] 1 手 = 100 股的最小交易单位
- [ ] T+1 交易制度
- [ ] 涨跌幅限制（主板 / 创业板 / 科创板 / ST / 北交所）
- [ ] 交易时间与交易日历
- [ ] 申报规则（价格档位、单笔上限、拆单）
- [ ] 费用模型（佣金 / 印花税 / 过户费）
- [ ] 除权除息处理
- [ ] 停牌与退市处理

### 11.2 数据质量模块

- [ ] 采集幂等与原始数据留档
- [ ] 结构校验（OHLC 关系、非负成交量）
- [ ] 数据去重
- [ ] 异常值标记
- [ ] 时效性保障（防前视）
- [ ] 数据源降级链
- [ ] 数据质量日报

### 11.3 分析流程模块

- [ ] 统一数据契约
- [ ] 特征提取与标准化
- [ ] 策略生成与信号一致性校验
- [ ] 回测验证
- [ ] 结果输出（表格与图表）
- [ ] 端到端可复现

### 11.4 智能体设计模块

- [ ] 角色定义与 Prompt 契约
- [ ] 能力边界（工具白名单）
- [ ] 交互逻辑与降级
- [ ] 任务调度（并发 / 限流 / 超时）
- [ ] 决策机制（先聚合后风控）
- [ ] 可观测性（token / 耗时 / finish_reason）

---

## 12. 贡献指南

1. Fork 本仓库并创建特性分支
2. 保持 `graph/schema.py` 中的 Pydantic 模型与 DB schema 同步
3. 新增数据接口请放在 `src/apis/` 下并在 `router.py` 暴露
4. 涉及统计口径的改动，请同步更新 `experiment/statistics.py` 并在 `supabase_stats.py` 中复跑校验

---

## 13. 致谢

本项目基于 [HKUSTDial/DeepFund](https://github.com/HKUSTDial/DeepFund) 开源项目进行 A 股特化改造。感谢原作者及论文
《Time Travel is Cheating: Going Live with DeepFund for Real-Time Fund Investment Benchmarking》的出色工作。

感谢 [AKShare](https://github.com/akfamily/akshare)、[LangGraph](https://github.com/langchain-ai/langgraph)、[Supabase](https://supabase.com/) 等开源项目。

---

> **⚠️ 免责声明**
>
> - 本系统产出的所有内容（信号、决策、报告、统计结果）均由 AI 自动生成，可能存在错误、偏差或幻觉
> - 本项目仅用于学术研究与技术验证，**不构成任何投资建议**
> - 历史回测与模拟结果不代表未来收益；投资决策请咨询持有中国证监会颁发资质的专业机构
> - 使用本工具进行实盘交易产生的任何投资损失，作者与贡献者不承担责任
> - 股市有风险，投资需谨慎
