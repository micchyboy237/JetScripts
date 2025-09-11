from jet.logger import logger
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
# GOAT

[GOAT](https://github.com/goat-sdk/goat) is the finance toolkit for AI agents.

Create agents that can:

- Send and receive payments
- Purchase physical and digital goods and services
- Engage in various investment strategies:
  - Earn yield
  - Bet on prediction markets
- Purchase crypto assets
- Tokenize any asset
- Get financial insights

### How it works
GOAT leverages blockchains, cryptocurrencies (such as stablecoins), and wallets as the infrastructure to enable agents to become economic actors:

1. Give your agent a [wallet](https://github.com/goat-sdk/goat/tree/main#chains-and-wallets)
2. Allow it to transact [anywhere](https://github.com/goat-sdk/goat/tree/main#chains-and-wallets)
3. Use more than [+200 tools](https://github.com/goat-sdk/goat/tree/main#tools)

See everything GOAT supports [here](https://github.com/goat-sdk/goat/tree/main#chains-and-wallets).

**Lightweight and extendable**
Different from other toolkits, GOAT is designed to be lightweight and extendable by keeping its core minimal and allowing you to install only the tools you need.

If you don't find what you need on our more than 200 integrations you can easily:

- Create your own plugin
- Integrate a new chain
- Integrate a new wallet
- Integrate a new agent framework

See how to do it [here](https://github.com/goat-sdk/goat/tree/main#-contributing).

## Installation and Setup

Check out our [quickstart](https://github.com/goat-sdk/goat/tree/main/python/examples/by-framework/langchain) to see how to set up and install GOAT.
"""
logger.info("# GOAT")

logger.info("\n\n[DONE]", bright=True)