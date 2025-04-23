import time
from datetime import datetime
from typing import Optional
from jet.llm.memory import Memory
from jet.llm.llm_types import Message
from jet.data import generate_unique_hash
from jet.db.chroma import ChromaClient, VectorItem, InitialDataEntry, SearchResult
from jet.logger import logger


class ChatHistory():
    def __init__(self, memory_id: str, messages: list[Message] = []):
        chats = self.filter_messages(messages)
        initial_data = [
            {
                "id": generate_unique_hash(),
                "document": chat['content'],
                "metadata": {
                    "role": chat['role'],
                    "created_at": time.time()
                },
            }
            for chat in chats
        ]
        settings = {
            "model": "llama3.1",
            "db_path": "data/vector_db",
            "related_memories_n": 10,
            "related_memories_dist": 0.7,
            "collections_settings": {
                "overwrite": False,
                "initial_data": initial_data,
            },
        }
        self.memory = Memory(memory_id=memory_id, settings=settings)
        self.collection_name = memory_id

    def filter_messages(self, messages: list[Message]):
        chats = [item for item in messages if item["role"]
                 in ["user", "assistant"] and item["content"]]
        return chats

    def store(self, messages: Message | list[Message]):
        messages = messages if isinstance(messages, list) else [messages]
        chats = self.filter_messages(messages)
        if chats:
            chat = chats[0]
            items = [
                {
                    "id": chat.get("id", generate_unique_hash()),
                    "document": chat["content"],
                    "embeddings": [],  # Placeholder: Add logic for generating vector embeddings
                    "metadata": {"role": chat['role'], "created_at": time.time()},
                }
            ]
            self.memory.chroma_client.upsert(items)

    def get(self):
        return self.memory.chroma_client.get()

    def search(self, texts: str | list[str]):
        return self.memory.chroma_client.search(texts=texts)

    def identify_memories(self, text: str, *args):
        return self.memory.identify_memories(text, *args)


if __name__ == "__main__":
    # messages: list[Message] = [
    #     {"role": "system", "content": "This is the system prompt for the conversation."},
    #     {"role": "user", "content": "Hello! How can I create a Python dictionary?"},
    #     {"role": "assistant", "content": "You can create a Python dictionary using curly braces, like this: `my_dict = {'key': 'value'}`."},
    #     {"role": "user", "content": "Thank you! Can you explain how to access values?"},
    #     {"role": "assistant",
    #         "content": "Sure! You can access values by using the key, like `my_dict['key']`."},
    #     {
    #         "role": "user",
    #         "content": "What is the mystery function on 5 and 6?"
    #     },
    #     {
    #         "role": "assistant",
    #         "content": "",
    #         "tool_calls": [
    #             {
    #                 "function": {
    #                     "name": "mystery",
    #                     "arguments": {
    #                         "a": 5,
    #                         "b": 6
    #                     }
    #                 }
    #             }
    #         ]
    #     },
    #     {
    #         "role": "tool",
    #         "content": -11
    #     }
    # ]
    # chat_history = ChatHistory(memory_id="history_test_1", messages=messages)
    # print(chat_history.get())
    # chat_history.store({
    #     "role": "user",
    #     "content": "Create a mystery function for 5 and 6."
    # })
    # print(chat_history.get())
    # search_results = chat_history.search(
    #     "Write a class given our code discussion")
    # print(search_results)

    chat_history = ChatHistory(memory_id="history_test_2")
    system = "You are searching for a new job. You will be provided some job posting details. Organize all information about the details as sectioned groups using markdown. Use --- as clear separator between sections."
    text = """Job Posting:

React Native App Developer (WFH)
 Bookmark

TYPE OF WORK
Full Time


SALARY
70,000PHP (negotiable)


HOURS PER WEEK
40


DATE POSTED
Dec 24, 2024

 JOB OVERVIEW
We are looking for a React Native App Developer interested in building performant mobile apps on both the iOS and Android platforms and desktop applications that works on both Windows and MacOS platform.

You will be responsible for architecting and building these applications, as well as coordinating with the teams responsible for other layers of the product infrastructure. Building a product is a highly collaborative effort, and as such, a strong team player with a commitment to perfection is required.

Job Requirements:
-Build pixel-perfect, buttery smooth UIs across both mobile platforms.
-Leverage native APIs for deep integrations with both platforms.
-Diagnose and fix bugs and performance bottlenecks for performance that feels native.
-Reach out to the open source community to encourage and help implement mission-critical software fixes—React Native moves fast and often breaks things.
-Maintain code and write automated tests to ensure the product is of the highest quality.
-Must own Mac/iOS device

Skills:
-Firm grasp of the JavaScript language and its nuances, including ES6+ syntax
-Knowledge of React-Native ecosystem
-Ability to write well-documented, clean Javascript code
-Rock solid at working with third-party dependencies and debugging dependency conflicts
-Familiarity with native build tools like XCode, Gradle, Swift
-Understanding of REST APIs, the document request model, and offline storage
-Experience with automated testing suites like Jest, Mocha or any other testing tools alternatives.
-Experience with version control like Git.

***Must be WFH ready
***Comfortable using time tracker
***Amenable to work on a FIXED schedule of 6AM-3PM // 7AM-4PM MNL time

Interested applicants may send their most updated CV with Skype ID and portfolio. Apply now!

 SKILL REQUIREMENT
React Native
ABOUT THE EMPLOYER
Contact Person: Anne Denise

Member since: January 22, 2024

Total Job Posts: 82
""".strip()
    results = chat_history.identify_memories(text, system)
    print(results)
