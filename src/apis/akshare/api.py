"""
AKShare API client implementation.
Link: https://github.com/akfamily/akshare
Docs: https://akshare.akfamily.xyz/
AKShare is free and does not require an API key.
"""

import re
import time
import pandas as pd
import requests
from datetime import datetime, timedelta, date as _date
from typing import Optional

import akshare as ak

from apis.common_model import OHLCVCandle, MediaNews
from .api_model import AStockFundamentals, AStockInsiderTrade, AStockMacroEconomic

DAILY_CANDLE_LOOKBACK_DAYS = 30

# ---- In-process cache for expensive full-market pulls ----
# Avoid re-downloading the same dataset for every ticker in a single workflow run.
_INSIDER_CACHE: dict = {"df": None, "timestamp": 0.0, "source": None}
_INSIDER_CACHE_TTL_SEC = 1800  # 30 minutes
_INSIDER_CACHE_EM_TTL_SEC = 3600  # em is slow; cache it longer

# ---- Eastmoney datacenter-web JSON 直连 ----
# RPT_EXECUTIVE_HOLD_DETAILS: 董监高及相关人员持股变动明细
# 全量 17 万+ 条 / 347 页 ≈ 210 s；过滤最近 N 天后 180 天 ≈ 6 s。
_EM_INSIDER_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
_EM_INSIDER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://data.eastmoney.com/",
}
_EM_INSIDER_WINDOW_DAYS = 30  # 默认拉最近30天；可按需调大
_EM_INSIDER_PAGE_SIZE = 100

# 东财底层英文列名 → 下游 pick_column 能识别的中文别名
_EM_INSIDER_COLUMN_MAP = {
    "SECURITY_CODE": "代码",
    "SECURITY_NAME": "名称",
    "CHANGE_DATE": "日期",
    "PERSON_NAME": "变动人",
    "DSE_PERSON_NAME": "董监高人员姓名",
    "POSITION_NAME": "职务",
    "PERSON_DSE_RELATION": "变动人与董监高关系",
    "CHANGE_SHARES": "变动股数",
    "AVERAGE_PRICE": "成交均价",
    "CHANGE_AMOUNT": "变动金额",
    "CHANGE_RATIO": "变动比例",
    "CHANGE_REASON": "变动原因",
    "CHANGE_AFTER_HOLDNUM": "变动后持股数",
    "BEGIN_HOLD_NUM": "开始时持有",
    "END_HOLD_NUM": "结束后持有",
    "HOLD_TYPE": "持股种类",
}

# em_fast 缓存（窗口有限，TTL 30min 足够）
_INSIDER_CACHE_EM_FAST_TTL_SEC = 1800

# 单标的查询缓存：key=(symbol, cutoff) -> (df, ts)
# 论文 Appendix D 的 insider_count=10 指"最近 10 笔董监高变动"，
# 与"最近 30 天"无关，因此按标的查询而非全市场窗口扫描。
_INSIDER_TICKER_CACHE: dict = {}
_INSIDER_TICKER_TTL_SEC = 1800

# Days of CCTV coverage to pull around trading_date so we catch both the
# policy event day and the day-after summaries without paying one round-trip
# per request.
_POLICY_CCTV_WINDOW_DAYS = 2

PRICE_COLUMNS = {
    "date": ["日期", "date", "Date"],
    "open": ["开盘", "open", "Open"],
    "high": ["最高", "high", "High"],
    "low": ["最低", "low", "Low"],
    "close": ["收盘", "close", "Close"],
    "volume": ["成交量", "volume", "Volume"],
}


class AKShareAPI:
    """AKShare API Wrapper for A-share market data."""
    def __init__(self):
        pass

    @staticmethod
    def _normalize_symbol(ticker: str) -> str:
        """Normalize an A-share ticker into the plain 6-digit code.

        Supports forms like 600519, 600519.SH, sh600519, SZ000001.
        """
        if not ticker:
            raise ValueError("Ticker must not be empty.")

        symbol = str(ticker).strip().upper()
        symbol = re.sub(r"^(SH|SZ|BJ)", "", symbol)
        symbol = re.sub(r"\.(SH|SZ|BJ)$", "", symbol)

        if not re.fullmatch(r"\d{6}", symbol):
            raise ValueError(f"Invalid A-share ticker: {ticker}")

        return symbol

    @staticmethod
    def _pick_column(df: pd.DataFrame, aliases: list) -> str:
        """Return the first existing column among the given aliases."""
        for alias in aliases:
            if alias in df.columns:
                return alias
        raise ValueError(
            f"None of the expected columns {aliases} found in {list(df.columns)}"
        )

    @staticmethod
    def _pick_optional_column(df: pd.DataFrame, aliases: list) -> Optional[str]:
        """Return the first existing column among the given aliases, or None."""
        for alias in aliases:
            if alias in df.columns:
                return alias
        return None

    @staticmethod
    def _to_float(value) -> Optional[float]:
        """Convert a value to float, returning None when not convertible."""
        try:
            if value is None or pd.isna(value):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_str(value) -> Optional[str]:
        """Convert a value to str, returning None for missing values."""
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        return str(value)

    @classmethod
    def _sanitize_record(cls, record: dict) -> dict:
        """Replace missing values in a record with None."""
        clean = {}
        for key, value in record.items():
            try:
                clean[key] = None if pd.isna(value) else value
            except (TypeError, ValueError):
                clean[key] = value
        return clean

    def _get_daily_candles(self, ticker: str, trading_date: datetime) -> list:
        """Get daily forward-adjusted candles for a ticker, filtered by trading_date."""
        symbol = self._normalize_symbol(ticker)
        start_date = (trading_date - timedelta(days=DAILY_CANDLE_LOOKBACK_DAYS)).strftime("%Y%m%d")
        end_date = trading_date.strftime("%Y%m%d")

        df = ak.stock_zh_a_hist_em(
            symbol=symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq",
        )

        if df is None or df.empty:
            return []

        date_col = self._pick_column(df, PRICE_COLUMNS["date"])
        open_col = self._pick_column(df, PRICE_COLUMNS["open"])
        high_col = self._pick_column(df, PRICE_COLUMNS["high"])
        low_col = self._pick_column(df, PRICE_COLUMNS["low"])
        close_col = self._pick_column(df, PRICE_COLUMNS["close"])
        volume_col = self._pick_column(df, PRICE_COLUMNS["volume"])

        df = df.dropna(subset=[close_col, volume_col])

        daily_candles = []
        for _, row in df.iterrows():
            candle_date = pd.to_datetime(row[date_col], errors="coerce")
            if pd.isna(candle_date) or candle_date > pd.Timestamp(trading_date):
                continue

            daily_candles.append(OHLCVCandle(
                date=candle_date.strftime("%Y-%m-%d"),
                open=float(row[open_col]),
                high=float(row[high_col]),
                low=float(row[low_col]),
                close=float(row[close_col]),
                volume=int(row[volume_col]),
            ))

        daily_candles.sort(key=lambda candle: candle.date)
        return daily_candles

    def get_daily_candles_df(self, ticker: str, trading_date: datetime) -> pd.DataFrame:
        """Convert daily candles into a DataFrame with datetime index and numeric columns."""
        daily_candles = self._get_daily_candles(ticker, trading_date)
        if not daily_candles:
            raise ValueError(f"No daily candles available for {ticker} up to {trading_date}")

        df = pd.DataFrame([candle.model_dump() for candle in daily_candles])

        df["Date"] = pd.to_datetime(df["date"])
        df.set_index("Date", inplace=True)
        df.drop("date", axis=1, inplace=True)

        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.sort_index(inplace=True)

        return df

    def get_last_close_price(self, ticker: str, trading_date: datetime) -> float:
        """Get the last close price on or before trading_date."""
        daily_candles = self._get_daily_candles(ticker, trading_date)
        if daily_candles:
            return daily_candles[-1].close

        return None

    def get_fundamentals(self, ticker: str, trading_date: datetime = None) -> AStockFundamentals:
        """Get company fundamentals for an A-share ticker."""
        symbol = self._normalize_symbol(ticker)

        info_df = ak.stock_individual_info_em(symbol=symbol)
        info = {}
        if info_df is not None and not info_df.empty:
            item_col = self._pick_column(info_df, ["item", "项目"])
            value_col = self._pick_column(info_df, ["value", "值"])
            info = dict(zip(info_df[item_col], info_df[value_col]))

        valuation = {}
        try:
            valuation_df = ak.stock_value_em(symbol=symbol)
            if valuation_df is not None and not valuation_df.empty:
                date_col = self._pick_column(valuation_df, ["数据日期"])
                valuation_df = valuation_df.copy()
                valuation_df["_dt"] = pd.to_datetime(valuation_df[date_col], errors="coerce")
                if trading_date is not None:
                    valuation_df = valuation_df[valuation_df["_dt"] <= pd.Timestamp(trading_date)]
                valuation_df = valuation_df.sort_values("_dt")
                if not valuation_df.empty:
                    valuation = valuation_df.iloc[-1].to_dict()
        except Exception:
            valuation = {}

        return AStockFundamentals(
            ticker=symbol,
            name=self._to_str(info.get("股票简称")),
            industry=self._to_str(info.get("行业")),
            listing_date=self._to_str(info.get("上市时间")),
            total_market_cap=self._to_float(info.get("总市值")),
            circulating_market_cap=self._to_float(info.get("流通市值")),
            total_shares=self._to_float(info.get("总股本")),
            circulating_shares=self._to_float(info.get("流通股")),
            pe_ttm=self._to_float(valuation.get("PE(TTM)")),
            pe_static=self._to_float(valuation.get("PE(静)")),
            pb=self._to_float(valuation.get("市净率")),
            peg=self._to_float(valuation.get("PEG值")),
            pcf=self._to_float(valuation.get("市现率")),
            ps=self._to_float(valuation.get("市销率")),
        )

    def get_news(self, ticker: str = None, trading_date: datetime = None, limit: int = None) -> list:
        """Get company news for an A-share ticker."""
        if not ticker:
            raise ValueError("Ticker is required to fetch company news.")

        symbol = self._normalize_symbol(ticker)
        df = ak.stock_news_em(symbol=symbol)
        if df is None or df.empty:
            return []

        title_col = self._pick_column(df, ["新闻标题", "title"])
        content_col = self._pick_column(df, ["新闻内容", "content"])
        time_col = self._pick_column(df, ["发布时间", "publish_time"])
        source_col = self._pick_column(df, ["文章来源", "publisher"])
        link_col = self._pick_column(df, ["新闻链接", "link"])

        df = df.copy()
        df["_publish_dt"] = pd.to_datetime(df[time_col], errors="coerce")
        if trading_date is not None:
            df = df[df["_publish_dt"] <= pd.Timestamp(trading_date)]
        df = df.sort_values("_publish_dt", ascending=False)

        news_list = []
        for _, row in df.iterrows():
            news_list.append(MediaNews(
                title=self._to_str(row[title_col]) or "",
                publish_time=self._to_str(row[time_col]) or "",
                publisher=self._to_str(row[source_col]) or "",
                link=self._to_str(row[link_col]),
                summary=self._to_str(row[content_col]),
            ))

        return news_list[:limit] if limit else news_list

    def get_policy_news(self, trading_date: datetime = None, limit: int = None) -> list:
        """Get macro/policy related news.

        Combines two sources:
        - stock_news_global_em (东方财富全球财经快讯): ~200 latest items only,
          fast and recent; has no history before the current day.
        - alternative_news_cctv (新闻联播文字稿): has data from 2016 to today,
          but only one day's coverage per call.

        For a historical trading_date, CCTV fills the gap.
        """
        frames = []

        # Source 1: 东方财富 global news (fast, recent-only).
        try:
            df_em = ak.stock_news_global_em()
            if df_em is not None and not df_em.empty:
                frames.append(("em", df_em))
        except Exception:
            pass

        # Source 2: CCTV news script — covers 2016→today, one day per call.
        # Pull a small window so we catch both the policy day and immediate
        # summaries, then the caller's trading_date filter narrows it down.
        if trading_date is not None:
            day_count = _POLICY_CCTV_WINDOW_DAYS
            for day_offset in range(day_count + 1):
                target_day = trading_date - timedelta(days=day_offset)
                try:
                    df_cctv = ak.alternative_news_cctv(date=target_day.strftime("%Y%m%d"))
                    if df_cctv is not None and not df_cctv.empty:
                        frames.append(("cctv", df_cctv))
                except Exception:
                    continue

        if not frames:
            return []

        news_list = []
        for source, df in frames:
            if source == "em":
                title_col = self._pick_column(df, ["标题", "title"])
                summary_col = self._pick_column(df, ["摘要", "summary"])
                time_col = self._pick_column(df, ["发布时间", "publish_time"])
                link_col = self._pick_column(df, ["链接", "link"])
                publisher = "东方财富"
            else:  # cctv
                title_col = self._pick_column(df, ["title", "标题"])
                summary_col = self._pick_column(df, ["content", "摘要"])
                time_col = self._pick_column(df, ["date", "发布时间"])
                link_col = None  # CCTV source doesn't expose a URL
                publisher = "央视新闻联播"

            df = df.copy()
            df["_publish_dt"] = pd.to_datetime(df[time_col], format="%Y%m%d", errors="coerce")
            if trading_date is not None:
                df = df[df["_publish_dt"] <= pd.Timestamp(trading_date)]
            df = df.sort_values("_publish_dt", ascending=False)

            for _, row in df.iterrows():
                news_list.append(MediaNews(
                    title=self._to_str(row[title_col]) or "",
                    publish_time=self._to_str(row[time_col]) or "",
                    publisher=publisher,
                    link=self._to_str(row[link_col]) if link_col else None,
                    summary=self._to_str(row[summary_col]),
                ))

        # Deduplicate by (publisher, title) before truncating.
        seen = set()
        deduped = []
        for item in news_list:
            key = (item.publisher, item.title)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)

        return deduped[:limit] if limit else deduped

    def get_insider_trades(self, ticker: str, trading_date: datetime = None, limit: int = None) -> list:
        """Get management shareholding changes for an A-share ticker.

        Three-tier source chain (fastest → most complete):

        1. **em_per_ticker**: 按标的直接查 Eastmoney datacenter-web JSON，
           取"最近 N 笔"变动（N=limit，默认 10），并以 trading_date 为上界。
           ~0.3 s/标的。**Preferred**：论文 ``insider_count`` 的语义是
           "最近 N 笔"，与日期窗口无关；蓝筹股变动稀疏（数年一次），
           仅按"最近 30 天"过滤会全部返回空。
        2. **em_fast**: 全市场近 30 天窗口（~6 s），作为兜底，
           覆盖窗口内确有变动的情形。
        3. **cninfo / em_full**: 已停用的慢路径（全市场 347 页 ≈ 210 s）。

        Each full-market pull is cached in-process so we only pay for it once
        per run across all tickers.
        """
        symbol = self._normalize_symbol(ticker)
        df = None

        # ---- Tier 1: 按标的查最近 N 笔（推荐，~0.3 s/标的）----
        df = self._load_insider_trades_em_by_ticker(
            symbol, limit=limit or 10, trading_date=trading_date
        )

        # ---- Tier 2: 全市场近 30 天窗口（兜底，~6 s）----
        if df is None or df.empty:
            df_em_fast = self._load_insider_trades_em_fast()
            df = self._filter_insider_by_ticker(df_em_fast, symbol) if df_em_fast is not None else None

        # ---- Tier 3: cninfo / em_full (disabled, slow) ----
        # if df is None or df.empty:
        #     df_cninfo = self._load_insider_trades_full(source="cninfo")
        #     df = self._filter_insider_by_ticker(df_cninfo, symbol) if df_cninfo is not None else None
        # if df is None or df.empty:
        #     df_em = self._load_insider_trades_full(source="em")
        #     df = self._filter_insider_by_ticker(df_em, symbol) if df_em is not None else None

        if df is None or df.empty:
            return []

        # Date-column mapping: cninfo uses 截止日期/公告日期, em uses 日期.
        cutoff_col = self._pick_optional_column(df, ["截止日期"])
        announce_col = self._pick_optional_column(df, ["公告日期"])
        if cutoff_col is not None:
            date_col = cutoff_col
            if announce_col is not None:
                df = df.copy()
                df[date_col] = df[date_col].fillna(df[announce_col])
        else:
            date_col = self._pick_column(df, ["日期", "变动日期"])

        executive_col = (
            self._pick_optional_column(df, ["董监高姓名", "董监高人员姓名"])
            or self._pick_optional_column(df, ["高管姓名", "变动人"])
        )
        if executive_col is None:
            executive_col = self._pick_column(df, ["姓名"])

        title_col = self._pick_optional_column(df, ["董监高职务", "职务"])
        relation_col = self._pick_optional_column(df, ["变动人与董监高关系", "与董监高关系", "关系"])
        shares_col = self._pick_column(df, ["变动数量", "变动股数"])
        price_col = self._pick_optional_column(df, ["成交均价"])
        ratio_col = self._pick_optional_column(df, ["变动比例"])
        shares_before_col = self._pick_optional_column(df, ["期初持股数量", "开始时持有"])
        shares_after_col = self._pick_optional_column(df, ["期末持股数量", "结束后持有", "变动后持股数"])
        reason_col = self._pick_optional_column(df, ["持股变动原因", "变动原因"])

        df = df.copy()
        df["_change_dt"] = pd.to_datetime(df[date_col], errors="coerce")
        if trading_date is not None:
            df = df[df["_change_dt"] <= pd.Timestamp(trading_date)]
        df = df.sort_values("_change_dt", ascending=False)

        trades = []
        for _, row in df.iterrows():
            trades.append(AStockInsiderTrade(
                ticker=symbol,
                transaction_date=self._to_str(row[date_col]) or "",
                executive=self._to_str(row[executive_col]) or "",
                executive_title=self._to_str(row[title_col]) if title_col else None,
                relationship=self._to_str(row[relation_col]) if relation_col else None,
                shares_changed=self._to_float(row[shares_col]),
                share_price=self._to_float(row[price_col]) if price_col else None,
                change_ratio=self._to_float(row[ratio_col]) if ratio_col else None,
                shares_before_change=self._to_float(row[shares_before_col]) if shares_before_col else None,
                shares_after_change=self._to_float(row[shares_after_col]) if shares_after_col else None,
                change_reason=self._to_str(row[reason_col]) if reason_col else None,
            ))

        return trades[:limit] if limit else trades

    @staticmethod
    def _filter_insider_by_ticker(df: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
        """Filter a full-market insider DataFrame down to one ticker."""
        if df is None or df.empty:
            return None
        code_col = None
        for cand in ["证券代码", "代码", "公司代码", "股票代码"]:
            if cand in df.columns:
                code_col = cand
                break
        if code_col is None:
            return None
        filtered = df[df[code_col].astype(str).str.zfill(6) == symbol]
        return filtered if not filtered.empty else None

    @classmethod
    def _load_insider_trades_em_by_ticker(
        cls,
        symbol: str,
        limit: int = 10,
        trading_date: datetime = None,
    ) -> Optional[pd.DataFrame]:
        """按单一标的拉取"最近 N 笔"董监高持股变动（RPT_EXECUTIVE_HOLD_DETAILS）。

        与全市场窗口扫描的区别：董监高变动是稀疏事件，蓝筹股可能数年才有一次
        （实测：600519 最近一次 2018-09-26、600036 2023-12-08、600028 2011-01-07）。
        旧实现只取"最近 30 天"全市场数据再按标的过滤，这些标的必然为空。
        论文 Appendix D 参数 ``insider_count=10`` 指"最近 10 笔变动"，
        因此这里按标的查询并取最近 N 笔，同时以 ``trading_date`` 为上界避免未来数据泄漏。
        """
        if not symbol:
            return None

        cutoff = None
        if trading_date is not None:
            cutoff = f"{pd.Timestamp(trading_date).strftime('%Y-%m-%d')} 23:59:59"

        cache_key = (symbol, cutoff, int(limit))
        cached = _INSIDER_TICKER_CACHE.get(cache_key)
        if cached is not None and (time.time() - cached[1]) < _INSIDER_TICKER_TTL_SEC:
            return cached[0]

        filter_sql = f'(SECURITY_CODE="{symbol}")'
        if cutoff:
            filter_sql += f"(CHANGE_DATE<='{cutoff}')"

        # 多取一些，便于下游按 trading_date 再过滤后仍能凑满 limit
        params = {
            "reportName": "RPT_EXECUTIVE_HOLD_DETAILS",
            "columns": "ALL",
            "quoteColumns": "",
            "filter": filter_sql,
            "pageSize": str(max(int(limit) * 3, 30)),
            "sortTypes": "-1",
            "sortColumns": "CHANGE_DATE",
            "source": "WEB",
            "client": "WEB",
            "pageNumber": "1",
        }

        try:
            r = requests.get(
                _EM_INSIDER_URL, params=params, headers=_EM_INSIDER_HEADERS, timeout=30
            )
            r.raise_for_status()
            payload = r.json()
        except Exception as e:
            from util.logger import logger
            logger.warning(f"em insider per-ticker failed for {symbol}: {e}")
            return None

        rows = (payload.get("result") or {}).get("data") or []
        if not rows:
            _INSIDER_TICKER_CACHE[cache_key] = (None, time.time())
            return None

        df = pd.DataFrame(rows)
        em_cols = [c for c in df.columns if c in _EM_INSIDER_COLUMN_MAP]
        df = df.rename(columns={c: _EM_INSIDER_COLUMN_MAP[c] for c in em_cols})
        _INSIDER_TICKER_CACHE[cache_key] = (df, time.time())
        return df

    @classmethod
    def _load_insider_trades_em_fast(cls, window_days: int = _EM_INSIDER_WINDOW_DAYS) -> Optional[pd.DataFrame]:
        """Fetch insider trades directly from Eastmoney datacenter-web JSON.

        RPT_EXECUTIVE_HOLD_DETAILS is the same dataset that
        ``ak.stock_hold_management_detail_em`` paginates through, but here we
        add a **CHANGE_DATE** filter so we only pull the last ``window_days``.

        Benchmark (2026-11-17, 5000 rows/page):
          window_days=30   → 2 pages,  ~1.0 s,   619 rows
          window_days=90   → 5 pages,  ~2.5 s, 2,152 rows
          window_days=180  → 11 pages, ~6.5 s, 5,048 rows
          window_days=365  → 24 pages,  ~16 s, 11,659 rows
          NO filter        → 347 pages, ~210 s, 173,253 rows (akshare slow path)

        The returned DataFrame is column-mapped so the downstream pick_column
        calls in ``get_insider_trades`` work without modification.
        """
        cache = _INSIDER_CACHE
        now = time.time()
        slot = "df_em_fast"
        ts_slot = "ts_em_fast"
        ttl = _INSIDER_CACHE_EM_FAST_TTL_SEC

        if cache.get(slot) is not None and (now - cache.get(ts_slot, 0.0)) < ttl:
            return cache[slot]

        from util.logger import logger

        since = (_date.today() - timedelta(days=window_days)).isoformat()
        filter_sql = f"(CHANGE_DATE>='{since}')"

        base_params = {
            "reportName": "RPT_EXECUTIVE_HOLD_DETAILS",
            "columns": "ALL",
            "quoteColumns": "",
            "filter": filter_sql,
            "pageSize": str(_EM_INSIDER_PAGE_SIZE),
            "sortTypes": "-1,1,1",
            "sortColumns": "CHANGE_DATE,SECURITY_CODE,PERSON_NAME",
            "source": "WEB",
            "client": "WEB",
        }

        try:
            r = requests.get(
                _EM_INSIDER_URL,
                params={**base_params, "pageNumber": "1"},
                headers=_EM_INSIDER_HEADERS,
                timeout=30,
            )
            r.raise_for_status()
            payload = r.json()
        except Exception as e:
            logger.warning(f"em_fast insider first page failed: {e}")
            # 不缓存失败结果：否则一次瞬时故障会让后续 30 分钟内所有标的都返回空
            cache[slot] = None
            cache[ts_slot] = 0.0
            return None

        result = payload.get("result") or {}
        pages = result.get("pages", 0) or 0
        if pages == 0 or not result.get("data"):
            cache[slot] = pd.DataFrame()
            cache[ts_slot] = now
            return None

        frames = [pd.DataFrame(result["data"])]
        for page in range(2, pages + 1):
            try:
                r = requests.get(
                    _EM_INSIDER_URL,
                    params={**base_params, "pageNumber": str(page)},
                    headers=_EM_INSIDER_HEADERS,
                    timeout=30,
                )
                r.raise_for_status()
                frames.append(pd.DataFrame((r.json() or {}).get("result", {}).get("data", [])))
            except Exception as e:
                logger.warning(f"em_fast insider page {page}/{pages} failed: {e}; carrying on.")
                break

        df = pd.concat(frames, ignore_index=True)
        if df.empty:
            cache[slot] = pd.DataFrame()
            cache[ts_slot] = now
            return None

        # ---- 列名映射（英文 → 中文别名）----
        em_cols = [c for c in df.columns if c in _EM_INSIDER_COLUMN_MAP]
        df = df.rename(columns={c: _EM_INSIDER_COLUMN_MAP[c] for c in em_cols})
        # 过滤后可能不再需要的列；保留其余列，下游按需 pick
        cache[slot] = df
        cache[ts_slot] = now
        return df

    @classmethod
    def _load_insider_trades_full(cls, source: str = "cninfo") -> Optional[pd.DataFrame]:
        """Load the full-market insider-trade dataset.

        source="cninfo" (default, fast): flat 增持 + 减持 calls (~4 s).
        source="em" (fallback, slow): Eastmoney pagination (~210 s).

        Both are cached in-process so we never pay for the same full-market
        download more than once per run.
        """
        cache = _INSIDER_CACHE
        now = time.time()

        # Cached under two separate slots; the slow em cache lives longer.
        slot = f"df_{source}"
        ts_slot = f"ts_{source}"
        ttl = _INSIDER_CACHE_EM_TTL_SEC if source == "em" else _INSIDER_CACHE_TTL_SEC

        if cache.get(slot) is not None and (now - cache.get(ts_slot, 0.0)) < ttl:
            return cache[slot]

        from util.logger import logger
        if source == "cninfo":
            frames = []
            try:
                for change_type in ("增持", "减持"):
                    df = ak.stock_hold_management_detail_cninfo(symbol=change_type)
                    if df is not None and not df.empty:
                        frames.append(df)
            except Exception as e:
                logger.error(f"cninfo insider fetch failed: {e}")

            if not frames:
                cache[slot] = pd.DataFrame()
                cache[ts_slot] = now
                return None
            combined = pd.concat(frames, ignore_index=True)
        else:  # em
            try:
                combined = ak.stock_hold_management_detail_em()
            except Exception as e:
                logger.error(f"Eastmoney insider fetch failed: {e}")
                cache[slot] = pd.DataFrame()
                cache[ts_slot] = now
                return None
            if combined is None or combined.empty:
                cache[slot] = pd.DataFrame()
                cache[ts_slot] = now
                return None

        cache[slot] = combined
        cache[ts_slot] = now
        return combined

    def get_economic_indicators(self) -> AStockMacroEconomic:
        """Get key China macroeconomic indicators."""
        return AStockMacroEconomic(
            gdp=self._fetch_macro(ak.macro_china_gdp, ["季度"]),
            cpi=self._fetch_macro(ak.macro_china_cpi, ["月份"]),
            pmi=self._fetch_macro(ak.macro_china_pmi, ["月份"]),
            lpr=self._fetch_macro(ak.interest_rate_lpr_em, ["日期", "TRADE_DATE", "月份"]),
            money_supply=self._fetch_macro(ak.macro_china_money_supply_em, ["月份"]),
        )

    @classmethod
    def _fetch_macro(cls, fetch_func, period_candidates: list, recent: int = 8) -> dict:
        """Fetch a macro indicator and keep its most recent records keyed by period."""
        try:
            df = fetch_func()
        except Exception:
            return {}

        if df is None or df.empty:
            return {}

        period_col = None
        for candidate in period_candidates:
            if candidate in df.columns:
                period_col = candidate
                break

        if period_col is not None:
            try:
                df = df.sort_values(period_col)
            except Exception:
                pass

        records = df.tail(recent).to_dict(orient="records")

        if period_col is None:
            return cls._sanitize_record(records[-1]) if records else {}

        result = {}
        for record in records:
            period = str(record.pop(period_col))
            result[period] = cls._sanitize_record(record)

        return result
