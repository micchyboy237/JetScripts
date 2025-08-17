from datetime import datetime
from jet.logger import CustomLogger
from mem0 import Memory
from typing import List, Dict
import anthropic
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")


# os.environ["OPENAI_API_KEY"] = "your_openai_api_key"  # needed for embedding model
# os.environ["ANTHROPIC_API_KEY"] = "your_anthropic_api_key"

class SupportChatbot:
    def __init__(self):
        self.config = {
            "llm": {
                "provider": "anthropic",
                "config": {
                    "model": "llama-3.2-3b-instruct",
                    "temperature": 0.1,
                    "max_tokens": 2000,
                },
            }
        }
#         self.client = anthropic.Client(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.memory = Memory.from_config(self.config)

        self.system_context = """
        You are a helpful customer support agent. Use the following guidelines:
        - Be polite and professional
        - Show empathy for customer issues
        - Reference past interactions when relevant
        - Maintain consistent information across conversations
        - If you're unsure about something, ask for clarification
        - Keep track of open issues and follow-ups
        """

    def store_customer_interaction(self, user_id: str, message: str, response: str, metadata: Dict = None):
        """Store customer interaction in memory."""
        if metadata is None:
            metadata = {}

        metadata["timestamp"] = datetime.now().isoformat()

        conversation = [{"role": "user", "content": message}, {"role": "assistant", "content": response}]

        self.memory.add(conversation, user_id=user_id, metadata=metadata)

    def get_relevant_history(self, user_id: str, query: str) -> List[Dict]:
        """Retrieve relevant past interactions."""
        return self.memory.search(
            query=query,
            user_id=user_id,
            limit=5,  # Adjust based on needs
        )

    def handle_customer_query(self, user_id: str, query: str) -> str:
        """Process customer query with context from past interactions."""

        relevant_history = self.get_relevant_history(user_id, query)

        context = "Previous relevant interactions:\n"
        for memory in relevant_history:
            context += f"Customer: {memory['memory']}\n"
            context += f"Support: {memory['memory']}\n"
            context += "---\n"

        prompt = f"""
        {self.system_context}

        {context}

        Current customer query: {query}

        Provide a helpful response that takes into account any relevant past interactions.
        """

        response = self.client.messages.create(
            model="llama-3.2-3b-instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.1,
        )

        self.store_customer_interaction(
            user_id=user_id, message=query, response=response, metadata={"type": "support_query"}
        )

        return response.content[0].text

chatbot = SupportChatbot()
user_id = "customer_bot"
logger.debug("Welcome to Customer Support! Type 'exit' to end the conversation.")

while True:
    query = input()
    logger.debug("Customer:", query)

    if query.lower() == "exit":
        logger.debug("Thank you for using our support service. Goodbye!")
        break

    response = chatbot.handle_customer_query(user_id, query)
    logger.debug("Support:", response, "\n\n")

logger.info("\n\n[DONE]", bright=True)