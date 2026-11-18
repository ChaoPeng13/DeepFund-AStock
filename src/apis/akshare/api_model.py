"""AKShare API models."""

from typing import Optional
from pydantic import BaseModel, Field

class AStockFundamentals(BaseModel):
    """A-share company fundamentals model.

    Only fields that AKShare can reliably provide are kept.
    """
    ticker: str
    # Basic info from stock_individual_info_em
    name: Optional[str] = None
    industry: Optional[str] = None
    listing_date: Optional[str] = None
    total_market_cap: Optional[float] = None
    circulating_market_cap: Optional[float] = None
    total_shares: Optional[float] = None
    circulating_shares: Optional[float] = None
    # Valuation metrics from stock_value_em
    pe_ttm: Optional[float] = None
    pe_static: Optional[float] = None
    pb: Optional[float] = None
    peg: Optional[float] = None
    pcf: Optional[float] = None
    ps: Optional[float] = None

class AStockInsiderTrade(BaseModel):
    """A-share management (insider) shareholding change model."""
    ticker: str
    transaction_date: str
    executive: str
    executive_title: Optional[str] = None
    relationship: Optional[str] = None
    shares_changed: Optional[float] = None
    share_price: Optional[float] = None
    change_ratio: Optional[float] = None
    shares_before_change: Optional[float] = None
    shares_after_change: Optional[float] = None
    change_reason: Optional[str] = None

class AStockMacroEconomic(BaseModel):
    """China macroeconomic indicator model.

    Each field holds recent records keyed by period, e.g.
    {"2025年第1季度": {"国内生产总值-绝对值": ...}}.
    """
    gdp: dict = Field(default_factory=dict)
    cpi: dict = Field(default_factory=dict)
    pmi: dict = Field(default_factory=dict)
    lpr: dict = Field(default_factory=dict)
    money_supply: dict = Field(default_factory=dict)
