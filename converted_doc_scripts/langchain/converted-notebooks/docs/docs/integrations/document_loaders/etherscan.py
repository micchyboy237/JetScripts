from jet.logger import logger
from langchain_community.document_loaders import EtherscanLoader
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
# Etherscan

>[Etherscan](https://docs.etherscan.io/)  is the leading blockchain explorer, search, API and analytics platform for Ethereum, 
a decentralized smart contracts platform.


## Overview

The `Etherscan` loader use `Etherscan API` to load transactions histories under specific account on `Ethereum Mainnet`.

You will need a `Etherscan api key` to proceed. The free api key has 5 calls per seconds quota.

The loader supports the following six functionalities:

* Retrieve normal transactions under specific account on Ethereum Mainet
* Retrieve internal transactions under specific account on Ethereum Mainet
* Retrieve erc20 transactions under specific account on Ethereum Mainet
* Retrieve erc721 transactions under specific account on Ethereum Mainet
* Retrieve erc1155 transactions under specific account on Ethereum Mainet
* Retrieve ethereum balance in wei under specific account on Ethereum Mainet


If the account does not have corresponding transactions, the loader will a list with one document. The content of document is ''.

You can pass different filters to loader to access different functionalities we mentioned above:

* "normal_transaction"
* "internal_transaction"
* "erc20_transaction"
* "eth_balance"
* "erc721_transaction"
* "erc1155_transaction"
The filter is default to normal_transaction

If you have any questions, you can access [Etherscan API Doc](https://etherscan.io/tx/0x0ffa32c787b1398f44303f731cb06678e086e4f82ce07cebf75e99bb7c079c77) or contact me via i@inevitable.tech.

All functions related to transactions histories are restricted 1000 histories maximum because of Etherscan limit. You can use the following parameters to find the transaction histories you need:

* offset: default to 20. Shows 20 transactions for one time
* page: default to 1. This controls pagination.
* start_block: Default to 0. The transaction histories starts from 0 block.
* end_block: Default to 99999999. The transaction histories starts from 99999999 block
* sort: "desc" or "asc". Set default to "desc" to get latest transactions.

## Setup
"""
logger.info("# Etherscan")

# %pip install --upgrade --quiet  langchain -q

etherscanAPIKey = "..."



os.environ["ETHERSCAN_API_KEY"] = etherscanAPIKey

"""
## Create a ERC20 transaction loader
"""
logger.info("## Create a ERC20 transaction loader")

account_address = "0x9dd134d14d1e65f84b706d6f205cd5b1cd03a46b"
loader = EtherscanLoader(account_address, filter="erc20_transaction")
result = loader.load()
eval(result[0].page_content)

"""
## Create a normal transaction loader with customized parameters
"""
logger.info("## Create a normal transaction loader with customized parameters")

loader = EtherscanLoader(
    account_address,
    page=2,
    offset=20,
    start_block=10000,
    end_block=8888888888,
    sort="asc",
)
result = loader.load()
result

logger.info("\n\n[DONE]", bright=True)