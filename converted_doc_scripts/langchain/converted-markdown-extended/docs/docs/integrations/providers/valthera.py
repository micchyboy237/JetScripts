from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_valthera.tools import ValtheraTool
from mocks import hubspot, posthog, snowflake  # Replace these with your actual connector implementations
from valthera.aggregator import DataAggregator
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
# Valthera

> [Valthera](https://github.com/valthera/valthera) is an open-source framework that empowers LLM Agents to drive meaningful, context-aware user engagement. It evaluates user motivation and ability in real time, ensuring that notifications and actions are triggered only when users are most receptive.
>
> **langchain-valthera** integrates Valthera with LangChain, enabling developers to build smarter, behavior-driven engagement systems that deliver personalized interactions.

## Installation and Setup

### Install langchain-valthera

Install the LangChain Valthera package via pip:
"""
logger.info("# Valthera")

pip install -U langchain-valthera

"""
Import the ValtheraTool:
"""
logger.info("Import the ValtheraTool:")


"""
### Example: Initializing the ValtheraTool for LangChain

This example shows how to initialize the ValtheraTool using a `DataAggregator` and configuration for motivation and ability scoring.
"""
logger.info("### Example: Initializing the ValtheraTool for LangChain")


data_aggregator = DataAggregator(
    connectors={
        "hubspot": hubspot(),
        "posthog": posthog(),
        "app_db": snowflake()
    }
)

valthera_tool = ValtheraTool(
    data_aggregator=data_aggregator,
    motivation_config=[
        {"key": "hubspot_lead_score", "weight": 0.30, "transform": lambda x: min(x, 100) / 100.0},
        {"key": "posthog_events_count_past_30days", "weight": 0.30, "transform": lambda x: min(x, 50) / 50.0},
        {"key": "hubspot_marketing_emails_opened", "weight": 0.20, "transform": lambda x: min(x / 10.0, 1.0)},
        {"key": "posthog_session_count", "weight": 0.20, "transform": lambda x: min(x / 5.0, 1.0)}
    ],
    ability_config=[
        {"key": "posthog_onboarding_steps_completed", "weight": 0.30, "transform": lambda x: min(x / 5.0, 1.0)},
        {"key": "posthog_session_count", "weight": 0.30, "transform": lambda x: min(x / 10.0, 1.0)},
        {"key": "behavior_complexity", "weight": 0.40, "transform": lambda x: 1 - (min(x, 5) / 5.0)}
    ]
)

logger.debug("✅ ValtheraTool successfully initialized for LangChain integration!")

"""
The langchain-valthera integration allows you to assess user behavior and decide on the best course of action for engagement, ensuring that interactions are both timely and relevant within your LangChain applications.
"""
logger.info("The langchain-valthera integration allows you to assess user behavior and decide on the best course of action for engagement, ensuring that interactions are both timely and relevant within your LangChain applications.")

logger.info("\n\n[DONE]", bright=True)