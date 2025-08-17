from jet.logger import CustomLogger
from mem0 import Memory
from openai import MLX
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: Customer Support AI Agent
---


You can create a personalized Customer Support AI Agent using Mem0. This guide will walk you through the necessary steps and provide the complete code to get you started.

## Overview

The Customer Support AI Agent leverages Mem0 to retain information across interactions, enabling a personalized and efficient support experience.

## Setup

Install the necessary packages using pip:
"""
logger.info("## Overview")

pip install openai mem0ai

"""
## Full Code Example

Below is the simplified code to create and interact with a Customer Support AI Agent using Mem0:
"""
logger.info("## Full Code Example")


# os.environ['OPENAI_API_KEY'] = 'sk-xxx'

class CustomerSupportAIAgent:
    def __init__(self):
        """
        Initialize the CustomerSupportAIAgent with memory configuration and MLX client.
        """
        config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "host": "localhost",
                    "port": 6333,
                }
            },
        }
        self.memory = Memory.from_config(config)
        self.client = MLX()
        self.app_id = "customer-support"

    def handle_query(self, query, user_id=None):
        """
        Handle a customer query and store the relevant information in memory.

        :param query: The customer query to handle.
        :param user_id: Optional user ID to associate with the memory.
        """
        stream = self.client.chat.completions.create(
            model="llama-3.2-3b-instruct", log_dir=f"{OUTPUT_DIR}/chats",
            stream=True,
            messages=[
                {"role": "system", "content": "You are a customer support AI agent."},
                {"role": "user", "content": query}
            ]
        )
        self.memory.add(query, user_id=user_id, metadata={"app_id": self.app_id})

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                logger.debug(chunk.choices[0].delta.content, end="")

    def get_memories(self, user_id=None):
        """
        Retrieve all memories associated with the given customer ID.

        :param user_id: Optional user ID to filter memories.
        :return: List of memories.
        """
        return self.memory.get_all(user_id=user_id)

support_agent = CustomerSupportAIAgent()

customer_id = "jane_doe"

support_agent.handle_query("I need help with my recent order. It hasn't arrived yet.", user_id=customer_id)

"""
### Fetching Memories

You can fetch all the memories at any point in time using the following code:
"""
logger.info("### Fetching Memories")

memories = support_agent.get_memories(user_id=customer_id)
for m in memories['results']:
    logger.debug(m['memory'])

"""
### Key Points

- **Initialization**: The CustomerSupportAIAgent class is initialized with the necessary memory configuration and MLX client setup.
- **Handling Queries**: The handle_query method sends a query to the AI and stores the relevant information in memory.
- **Retrieving Memories**: The get_memories method fetches all stored memories associated with a customer.

### Conclusion

As the conversation progresses, Mem0's memory automatically updates based on the interactions, providing a continuously improving personalized support experience.
"""
logger.info("### Key Points")

logger.info("\n\n[DONE]", bright=True)