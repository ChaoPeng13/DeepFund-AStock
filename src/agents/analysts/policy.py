from graph.constants import AgentKey
from llm.prompt import POLICY_PROMPT
from graph.schema import FundState, AnalystSignal
from llm.inference import agent_call
from apis.router import Router, APISource
from util.db_helper import get_db
from util.logger import logger

# thresholds
thresholds = {
    "news_count": 10,
}

def policy_agent(state: FundState):
    """policy specialist analyzing market news to provide a signal."""
    agent_name = AgentKey.POLICY
    ticker = state["ticker"]
    trading_date = state["trading_date"]
    llm_config = state["llm_config"]
    portfolio_id = state["portfolio"].id

    # Get db instance
    db = get_db()
    
    logger.log_agent_status(agent_name, ticker, "Fetching policy related news")
    
    # Get the policy news
    router = Router(APISource.AKSHARE)
    try:
        policy_news = router.get_a_stock_policy_news(
            trading_date=trading_date,
            news_count=thresholds["news_count"]
        )
    except Exception as e:
        logger.error(f"Failed to fetch policy news for {ticker}: {e}")
        return state

    # Analyze news sentiment via LLM
    policy_news_dict = [m.model_dump_json() for m in policy_news]
    prompt = POLICY_PROMPT.format(policy_news=policy_news_dict)

    # Get LLM signal
    signal = agent_call(
        prompt=prompt,
        llm_config=llm_config,
        pydantic_model=AnalystSignal,
        agent=f"{agent_name} ({ticker})",
    )

    # save signal
    logger.log_signal(agent_name, ticker, signal)
    db.save_signal(portfolio_id, agent_name, ticker, prompt, signal)
    
    return {"analyst_signals": [signal]}
