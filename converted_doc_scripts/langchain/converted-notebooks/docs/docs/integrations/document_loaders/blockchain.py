from jet.logger import logger
from langchain_community.document_loaders.blockchain import (
BlockchainDocumentLoader,
BlockchainType,
)
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
# Blockchain

The intention of this notebook is to provide a means of testing functionality in the Langchain Document Loader for Blockchain.

Initially this Loader supports:

*   Loading NFTs as Documents from NFT Smart Contracts (ERC721 and ERC1155)
*   Ethereum Mainnnet, Ethereum Testnet, Polygon Mainnet, Polygon Testnet (default is eth-mainnet)
*   Alchemy's getNFTsForCollection API

It can be extended if the community finds value in this loader.  Specifically:

*   Additional APIs can be added (e.g. Tranction-related APIs)

This Document Loader Requires:

*   A free [Alchemy API Key](https://www.alchemy.com/)

The output takes the following format:

- pageContent= Individual NFT
- metadata=\{'source': '0x1a92f7381b9f03921564a437210bb9396471050c', 'blockchain': 'eth-mainnet', 'tokenId': '0x15'\}

## Load NFTs into Document Loader
"""
logger.info("# Blockchain")

alchemyApiKey = "..."

"""
### Option 1: Ethereum Mainnet (default BlockchainType)
"""
logger.info("### Option 1: Ethereum Mainnet (default BlockchainType)")


contractAddress = "0xbc4ca0eda7647a8ab7c2061c2e118a18a936f13d"  # Bored Ape Yacht Club contract address

blockchainType = BlockchainType.ETH_MAINNET  # default value, optional parameter

blockchainLoader = BlockchainDocumentLoader(
    contract_address=contractAddress, api_key=alchemyApiKey
)

nfts = blockchainLoader.load()

nfts[:2]

"""
### Option 2: Polygon Mainnet
"""
logger.info("### Option 2: Polygon Mainnet")

contractAddress = (
    "0x448676ffCd0aDf2D85C1f0565e8dde6924A9A7D9"  # Polygon Mainnet contract address
)

blockchainType = BlockchainType.POLYGON_MAINNET

blockchainLoader = BlockchainDocumentLoader(
    contract_address=contractAddress,
    blockchainType=blockchainType,
    api_key=alchemyApiKey,
)

nfts = blockchainLoader.load()

nfts[:2]

logger.info("\n\n[DONE]", bright=True)