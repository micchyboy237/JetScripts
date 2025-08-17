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
title: Personalized AI Tutor
---

You can create a personalized AI Tutor using Mem0. This guide will walk you through the necessary steps and provide the complete code to get you started.

## Overview

The Personalized AI Tutor leverages Mem0 to retain information across interactions, enabling a tailored learning experience. By integrating with MLX's GPT-4 model, the tutor can provide detailed and context-aware responses to user queries.

## Setup
Before you begin, ensure you have the required dependencies installed. You can install the necessary packages using pip:
"""
logger.info("## Overview")

pip install openai mem0ai

"""
## Full Code Example

Below is the complete code to create and interact with a Personalized AI Tutor using Mem0:
"""
logger.info("## Full Code Example")


# os.environ['OPENAI_API_KEY'] = 'sk-xxx'

client = MLX()

class PersonalAITutor:
    def __init__(self):
        """
        Initialize the PersonalAITutor with memory configuration and MLX client.
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
        self.client = client
        self.app_id = "app-1"

    def ask(self, question, user_id=None):
        """
        Ask a question to the AI and store the relevant facts in memory

        :param question: The question to ask the AI.
        :param user_id: Optional user ID to associate with the memory.
        """
        response = self.client.responses.create(
            model="llama-3.2-3b-instruct", log_dir=f"{OUTPUT_DIR}/chats",
            instructions="You are a personal AI Tutor.",
            input=question,
            stream=True
        )

        self.memory.add(question, user_id=user_id, metadata={"app_id": self.app_id})

        for event in response:
            if event.type == "response.output_text.delta":
                logger.debug(event.delta, end="")

    def get_memories(self, user_id=None):
        """
        Retrieve all memories associated with the given user ID.

        :param user_id: Optional user ID to filter memories.
        :return: List of memories.
        """
        return self.memory.get_all(user_id=user_id)

ai_tutor = PersonalAITutor()

user_id = "john_doe"

ai_tutor.ask("I am learning introduction to CS. What is queue? Briefly explain.", user_id=user_id)

"""
### Fetching Memories

You can fetch all the memories at any point in time using the following code:
"""
logger.info("### Fetching Memories")

memories = ai_tutor.get_memories(user_id=user_id)
for m in memories['results']:
    logger.debug(m['memory'])

"""
### Key Points

- **Initialization**: The PersonalAITutor class is initialized with the necessary memory configuration and MLX client setup.
- **Asking Questions**: The ask method sends a question to the AI and stores the relevant information in memory.
- **Retrieving Memories**: The get_memories method fetches all stored memories associated with a user.

### Conclusion

As the conversation progresses, Mem0's memory automatically updates based on the interactions, providing a continuously improving personalized learning experience. This setup ensures that the AI Tutor can offer contextually relevant and accurate responses, enhancing the overall educational process.
"""
logger.info("### Key Points")

logger.info("\n\n[DONE]", bright=True)