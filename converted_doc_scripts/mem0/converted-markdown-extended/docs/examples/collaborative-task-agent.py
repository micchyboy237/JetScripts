from collections import defaultdict
from datetime import datetime
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
title: Multi-User Collaboration with Mem0
---

## Overview

Build a multi-user collaborative chat or task management system with Mem0. Each message is attributed to its author, and all messages are stored in a shared project space. Mem0 makes it easy to track contributions, sort and group messages, and collaborate in real time.

## Setup

Install the required packages:
"""
logger.info("## Overview")

pip install openai mem0ai

"""
## Full Code Example
"""
logger.info("## Full Code Example")


# os.environ["OPENAI_API_KEY"] = "sk-your-key"

RUN_ID = "project-demo"

mem = Memory()

class CollaborativeAgent:
    def __init__(self, run_id):
        self.run_id = run_id
        self.mem = mem

    def add_message(self, role, name, content):
        msg = {"role": role, "name": name, "content": content}
        self.mem.add([msg], run_id=self.run_id, infer=False)

    def brainstorm(self, prompt):
        memories = self.mem.search(prompt, run_id=self.run_id, limit=5)["results"]
        context = "\n".join(f"- {m['memory']} (by {m.get('actor_id', 'Unknown')})" for m in memories)
        client = MLX()
        messages = [
            {"role": "system", "content": "You are a helpful project assistant."},
            {"role": "user", "content": f"Prompt: {prompt}\nContext:\n{context}"}
        ]
        reply = client.chat.completions.create(
            model="llama-3.2-3b-instruct",
            messages=messages
        ).choices[0].message.content.strip()
        self.add_message("assistant", "assistant", reply)
        return reply

    def get_all_messages(self):
        return self.mem.get_all(run_id=self.run_id)["results"]

    def print_sorted_by_time(self):
        messages = self.get_all_messages()
        messages.sort(key=lambda m: m.get('created_at', ''))
        logger.debug("\n--- Messages (sorted by time) ---")
        for m in messages:
            who = m.get("actor_id") or "Unknown"
            ts = m.get('created_at', 'Timestamp N/A')
            try:
                dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                ts_fmt = dt.strftime('%Y-%m-%d %H:%M:%S')
            except Exception:
                ts_fmt = ts
            logger.debug(f"[{ts_fmt}] [{who}] {m['memory']}")

    def print_grouped_by_actor(self):
        messages = self.get_all_messages()
        grouped = defaultdict(list)
        for m in messages:
            grouped[m.get("actor_id") or "Unknown"].append(m)
        logger.debug("\n--- Messages (grouped by actor) ---")
        for actor, mems in grouped.items():
            logger.debug(f"\n=== {actor} ===")
            for m in mems:
                ts = m.get('created_at', 'Timestamp N/A')
                try:
                    dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    ts_fmt = dt.strftime('%Y-%m-%d %H:%M:%S')
                except Exception:
                    ts_fmt = ts
                logger.debug(f"[{ts_fmt}] {m['memory']}")

"""
## Usage
"""
logger.info("## Usage")

agent = CollaborativeAgent(RUN_ID)
agent.add_message("user", "alice", "Let's list tasks for the new landing page.")
agent.add_message("user", "bob", "I'll own the hero section copy.")
agent.add_message("user", "carol", "I'll choose product screenshots.")

logger.debug("\nAssistant reply:\n", agent.brainstorm("What are the current open tasks?"))

agent.print_sorted_by_time()

agent.print_grouped_by_actor()

"""
## Key Points

- Each message is attributed to a user or agent (actor)
- All messages are stored in a shared project space (`run_id`)
- You can sort messages by time, group by actor, and format timestamps for clarity
- Mem0 makes it easy to build collaborative, attributed chat/task systems

## Conclusion

Mem0 enables fast, transparent collaboration for teams and agents, with full attribution, flexible memory search, and easy message organization.
"""
logger.info("## Key Points")

logger.info("\n\n[DONE]", bright=True)