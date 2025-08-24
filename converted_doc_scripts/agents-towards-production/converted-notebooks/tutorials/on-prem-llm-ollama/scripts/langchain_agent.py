from jet.logger import CustomLogger
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOllama
from typing import Dict, List
import asyncio
import os
import requests
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# LangChain Agent with Ollama

## 💡 This agent shows how to:
• Build a simple but complete agent using LangChain and Ollama

• Replace MLX models with Ollama in LangChain

• Chain multiple LLM calls together

• Build structured workflows

• Handle different types of analysis tasks

• Quick and clear way to showcase the agent’s ability to interpret and analyze various kinds of textual input.
"""
logger.info("# LangChain Agent with Ollama")


class SimpleAnalysisAgent:
    """A simple agent that analyzes text and provides insights."""

    def __init__(self, model_name: str = "llama3.1:8b"):
        if not self.check_ollama_model(model_name):
            logger.debug(f"❌ Model {model_name} not found. Try: ollama pull {model_name}")
            raise ValueError(f"Model {model_name} not available")

        self.llm = ChatOllama(model="qwen3-1.7b-4bit")
        self.conversation_history = []

    def classify_text(self, text: str) -> str:
        """Classify the type of text."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Classify this text as one of: news, blog, email, code, academic, or other. Respond with just the category."),
            ("human", "{text}")
        ])

        chain = prompt | self.llm
        result = chain.invoke({"text": text[:500]})  # Limit length
        return result.content.strip().lower()

    def extract_key_points(self, text: str) -> List[str]:
        """Extract key points from text."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Extract 3-5 key points from this text. Return as a simple numbered list."),
            ("human", "{text}")
        ])

        chain = prompt | self.llm
        result = chain.invoke({"text": text})

        lines = result.content.strip().split('\n')
        points = [line.strip() for line in lines if line.strip() and any(c.isdigit() for c in line[:3])]
        return points[:5]  # Limit to 5 points

    def summarize(self, text: str) -> str:
        """Create a summary of the text."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Summarize this text in 2-3 sentences. Be concise and clear."),
            ("human", "{text}")
        ])

        chain = prompt | self.llm
        result = chain.invoke({"text": text})
        return result.content.strip()

    def analyze_text(self, text: str) -> Dict:
        """Complete analysis of text."""
        logger.debug(f"🔍 Analyzing text ({len(text)} characters)...")

        logger.debug("  📊 Classifying...")
        category = self.classify_text(text)

        logger.debug("  🔑 Extracting key points...")
        key_points = self.extract_key_points(text)

        logger.debug("  📝 Summarizing...")
        summary = self.summarize(text)

        return {
            "category": category,
            "key_points": key_points,
            "summary": summary,
            "length": len(text)
        }

    @staticmethod
    def check_ollama_model(model_name: str) -> bool:
        try:
            response = requests.get("http://localhost:11434/api/tags")
            models = [m["name"] for m in response.json().get("models", [])]
            return model_name in models
        except:
            return False

"""
# Demo agent:

The demo_agent() function is used to demonstrate the text analysis agent in action. A set of predefined sample texts is provided, each representing different content types (e.g., a news article or a code snippet).

The function processes each sample and displays the results, including:

• Detected content category

• Summary of the text

• List of extracted key points
"""
logger.info("# Demo agent:")

def demo_agent():
    """Demonstrate the agent with sample texts."""
    logger.debug("🤖 LangChain Agent with Ollama Demo")
    logger.debug("=" * 40)

    try:
        agent = SimpleAnalysisAgent()
        logger.debug("✅ Agent initialized successfully\n")
    except ValueError as e:
        logger.debug(e)
        return

    samples = [
        {
            "name": "Tech News",
            "text": """
            Apple announced today that its new iPhone 15 will feature USB-C charging,
            marking a significant shift from the Lightning connector. The change comes
            after pressure from the European Union's new charging regulations. The new
            phones will also include improved cameras and faster processors. Industry
            analysts expect this to boost sales significantly in the next quarter.
            """
        },
        {
            "name": "Code Comment",
            "text": """
            def process_data(input_list):
                if not input_list:
                    return 0
                return sum(input_list) / len(input_list)
            """
        }
    ]

    for sample in samples:
        logger.debug(f"📋 Sample: {sample['name']}")
        logger.debug("-" * 30)

        result = agent.analyze_text(sample["text"].strip())

        logger.debug(f"Category: {result['category']}")
        logger.debug(f"Summary: {result['summary']}")
        logger.debug("Key Points:")
        for point in result['key_points']:
            logger.debug(f"  • {point}")
        logger.debug()

demo_agent()
logger.debug("\n✅ Done!")

"""
# interactive mode:

The interactive_mode() function offers a simple command-line interface for real-time testing of a text analysis agent.

It allows users to input freeform text and receive immediate feedback which includes a category classification, a summary, and key points extracted from the content.

You can experiment with different types of input to see how the model interprets and summarizes information.
"""
logger.info("# interactive mode:")

def interactive_mode():
    """Interactive mode for testing with your own text."""
    logger.debug("🔄 Interactive Mode")
    logger.debug("=" * 40)

    try:
        agent = SimpleAnalysisAgent()
    except ValueError as e:
        logger.debug(e)
        return

    logger.debug("Enter text to analyze (or 'quit' to exit):")

    while True:
        text = input("\n> ")

        if text.lower() in ['quit', 'exit', 'q']:
            break

        if len(text.strip()) < 10:
            logger.debug("Please enter more text (at least 10 characters)")
            continue

        result = agent.analyze_text(text)

        logger.debug(f"\n📊 Results:")
        logger.debug(f"Category: {result['category']}")
        logger.debug(f"Summary: {result['summary']}")
        if result['key_points']:
            logger.debug("Key Points:")
            for point in result['key_points']:
                logger.debug(f"  • {point}")

interactive_mode()

logger.info("\n\n[DONE]", bright=True)