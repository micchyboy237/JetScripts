from MintbaseLoader import MintbaseDocumentLoader
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
# Near Blockchain

The intention of this notebook is to provide a means of testing functionality in the Langchain Document Loader for Near Blockchain.

Initially this Loader supports:

*   Loading NFTs as Documents from NFT Smart Contracts (NEP-171 and NEP-177)
*   Near Mainnnet, Near Testnet (default is mainnet)
*   Mintbase's Graph API

It can be extended if the community finds value in this loader.  Specifically:

*   Additional APIs can be added (e.g. Tranction-related APIs)

This Document Loader Requires:

*   A free [Mintbase API Key](https://docs.mintbase.xyz/dev/mintbase-graph/)

The output takes the following format:

- pageContent= Individual NFT
- metadata=\{'source': 'nft.yearofchef.near', 'blockchain': 'mainnet', 'tokenId': '1846'\}

## Load NFTs into Document Loader
"""
logger.info("# Near Blockchain")

mintbaseApiKey = "..."

"""
### Option 1: Ethereum Mainnet (default BlockchainType)
"""
logger.info("### Option 1: Ethereum Mainnet (default BlockchainType)")


contractAddress = "nft.yearofchef.near"  # Year of chef contract address


blockchainLoader = MintbaseDocumentLoader(
    contract_address=contractAddress, blockchain_type="mainnet"
)

nfts = blockchainLoader.load()

logger.debug(nfts[:1])

for doc in blockchainLoader.lazy_load():
    logger.debug()
    logger.debug(type(doc))
    logger.debug(doc)

logger.info("\n\n[DONE]", bright=True)