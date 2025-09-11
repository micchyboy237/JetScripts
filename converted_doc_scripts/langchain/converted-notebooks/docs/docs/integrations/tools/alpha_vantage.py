from jet.logger import logger
from langchain_community.utilities.alpha_vantage import AlphaVantageAPIWrapper
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
# Alpha Vantage

>[Alpha Vantage](https://www.alphavantage.co) Alpha Vantage provides realtime and historical financial market data through a set of powerful and developer-friendly data APIs and spreadsheets.

Generate the `ALPHAVANTAGE_API_KEY` [at their website](https://www.alphavantage.co/support/#api-key)
.

Use the ``AlphaVantageAPIWrapper`` to get currency exchange rates.
"""
logger.info("# Alpha Vantage")

# import getpass

# os.environ["ALPHAVANTAGE_API_KEY"] = getpass.getpass()


alpha_vantage = AlphaVantageAPIWrapper()
alpha_vantage._get_exchange_rate("USD", "JPY")

"""
The `_get_time_series_daily` method returns the date, daily open, daily high, daily low, daily close, and daily volume of the global equity specified, covering the 100 latest data points.
"""
logger.info("The `_get_time_series_daily` method returns the date, daily open, daily high, daily low, daily close, and daily volume of the global equity specified, covering the 100 latest data points.")

alpha_vantage._get_time_series_daily("IBM")

"""
The `_get_time_series_weekly` method returns the last trading day of the week, weekly open, weekly high, weekly low, weekly close, and weekly volume of the global equity specified, covering 20+ years of historical data.
"""
logger.info("The `_get_time_series_weekly` method returns the last trading day of the week, weekly open, weekly high, weekly low, weekly close, and weekly volume of the global equity specified, covering 20+ years of historical data.")

alpha_vantage._get_time_series_weekly("IBM")

"""
The `_get_quote_endpoint` method is a lightweight alternative to the time series APIs and returns the latest price and volume info for the specified symbol.
"""
logger.info("The `_get_quote_endpoint` method is a lightweight alternative to the time series APIs and returns the latest price and volume info for the specified symbol.")

alpha_vantage._get_quote_endpoint("IBM")

"""
The `search_symbol` method returns a list of symbols and the matching company information based on the text entered.
"""
logger.info("The `search_symbol` method returns a list of symbols and the matching company information based on the text entered.")

alpha_vantage.search_symbols("IB")

"""
The `_get_market_news_sentiment` method returns live and historical market news sentiment for a given asset.
"""
logger.info("The `_get_market_news_sentiment` method returns live and historical market news sentiment for a given asset.")

alpha_vantage._get_market_news_sentiment("IBM")

"""
The `_get_top_gainers_losers` method returns the top 20 gainers, losers and most active stocks in the US market.
"""
logger.info("The `_get_top_gainers_losers` method returns the top 20 gainers, losers and most active stocks in the US market.")

alpha_vantage._get_top_gainers_losers()

"""
The `run` method of the wrapper takes the following parameters: from_currency, to_currency. 

It Gets the currency exchange rates for the given currency pair.
"""
logger.info("The `run` method of the wrapper takes the following parameters: from_currency, to_currency.")

alpha_vantage.run("USD", "JPY")

logger.info("\n\n[DONE]", bright=True)