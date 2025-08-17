import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
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
title: Contextual Memory Creation
icon: "square-plus"
iconType: "solid"
description: "Add messages with automatic context management - no manual history tracking required"
---

## What is Contextual Memory Creation?

Contextual memory creation automatically manages message history for you, so you can focus on building great AI experiences instead of tracking interactions manually. Simply send new messages, and Mem0 handles the context automatically.

<CodeGroup>
"""
logger.info("## What is Contextual Memory Creation?")

messages = [
    {"role": "user", "content": "I love Italian food, especially pasta"},
    {"role": "assistant", "content": "Great! I'll remember your preference for Italian cuisine."}
]

client.add(messages, user_id="user123", version="v2")

"""

"""

messages = [
    {"role": "user", "content": "I love Italian food, especially pasta"},
    {"role": "assistant", "content": "Great! I'll remember your preference for Italian cuisine."}
]

async def run_async_code_ea9d2566():
    await client.add(messages, { user_id: "user123", version: "v2" })
    return 
 = asyncio.run(run_async_code_ea9d2566())
logger.success(format_json())

"""
</CodeGroup>

## Why Use Contextual Memory Creation?

- **Simple**: Send only new messages, no manual history tracking
- **Efficient**: Smaller payloads and faster processing
- **Automatic**: Context management handled by Mem0
- **Reliable**: No risk of missing interaction history
- **Scalable**: Works seamlessly as your application grows

## How It Works

### Basic Usage

<CodeGroup>
"""
logger.info("## Why Use Contextual Memory Creation?")

messages1 = [
    {"role": "user", "content": "Hi, I'm Sarah from New York"},
    {"role": "assistant", "content": "Hello Sarah! Nice to meet you."}
]
client.add(messages1, user_id="sarah", version="v2")

messages2 = [
    {"role": "user", "content": "I'm planning a trip to Italy next month"},
    {"role": "assistant", "content": "How exciting! Italy is beautiful this time of year."}
]
client.add(messages2, user_id="sarah", version="v2")

"""

"""

messages1 = [
    {"role": "user", "content": "Hi, I'm Sarah from New York"},
    {"role": "assistant", "content": "Hello Sarah! Nice to meet you."}
]
async def run_async_code_1d5dda43():
    await client.add(messages1, { user_id: "sarah", version: "v2" })
    return 
 = asyncio.run(run_async_code_1d5dda43())
logger.success(format_json())

messages2 = [
    {"role": "user", "content": "I'm planning a trip to Italy next month"},
    {"role": "assistant", "content": "How exciting! Italy is beautiful this time of year."}
]
async def run_async_code_a0e3e8b1():
    await client.add(messages2, { user_id: "sarah", version: "v2" })
    return 
 = asyncio.run(run_async_code_a0e3e8b1())
logger.success(format_json())

"""
</CodeGroup>

## Organization Strategies

Choose the right approach based on your application's needs:

### User-Level Memories (`user_id` only)

Best for: Personal preferences, profile information, long-term user data

<CodeGroup>
"""
logger.info("## Organization Strategies")

messages = [
    {"role": "user", "content": "I'm allergic to nuts and dairy"},
    {"role": "assistant", "content": "I've noted your allergies for future reference."}
]

client.add(messages, user_id="user123", version="v2")

"""

"""

messages = [
    {"role": "user", "content": "I'm allergic to nuts and dairy"},
    {"role": "assistant", "content": "I've noted your allergies for future reference."}
]

async def run_async_code_ea9d2566():
    await client.add(messages, { user_id: "user123", version: "v2" })
    return 
 = asyncio.run(run_async_code_ea9d2566())
logger.success(format_json())

"""
</CodeGroup>

### Session-Specific Memories (`user_id` + `run_id`)

Best for: Task-specific context, separate interaction threads, project-based sessions

<CodeGroup>
"""
logger.info("### Session-Specific Memories (`user_id` + `run_id`)")

messages1 = [
    {"role": "user", "content": "I want to plan a 5-day trip to Tokyo"},
    {"role": "assistant", "content": "Perfect! Let's plan your Tokyo adventure."}
]
client.add(messages1, user_id="user123", run_id="tokyo-trip-2024", version="v2")

messages2 = [
    {"role": "user", "content": "I prefer staying near Shibuya"},
    {"role": "assistant", "content": "Great choice! Shibuya is very convenient."}
]
client.add(messages2, user_id="user123", run_id="tokyo-trip-2024", version="v2")

work_messages = [
    {"role": "user", "content": "Let's discuss the Q4 marketing strategy"},
    {"role": "assistant", "content": "Sure! What are your main goals for Q4?"}
]
client.add(work_messages, user_id="user123", run_id="q4-marketing", version="v2")

"""

"""

messages1 = [
    {"role": "user", "content": "I want to plan a 5-day trip to Tokyo"},
    {"role": "assistant", "content": "Perfect! Let's plan your Tokyo adventure."}
]
async def run_async_code_0205110b():
    await client.add(messages1, { user_id: "user123", run_id: "tokyo-trip-2024", version: "v2" })
    return 
 = asyncio.run(run_async_code_0205110b())
logger.success(format_json())

messages2 = [
    {"role": "user", "content": "I prefer staying near Shibuya"},
    {"role": "assistant", "content": "Great choice! Shibuya is very convenient."}
]
async def run_async_code_3793f431():
    await client.add(messages2, { user_id: "user123", run_id: "tokyo-trip-2024", version: "v2" })
    return 
 = asyncio.run(run_async_code_3793f431())
logger.success(format_json())

workMessages = [
    {"role": "user", "content": "Let's discuss the Q4 marketing strategy"},
    {"role": "assistant", "content": "Sure! What are your main goals for Q4?"}
]
async def run_async_code_551060f9():
    await client.add(workMessages, { user_id: "user123", run_id: "q4-marketing", version: "v2" })
    return 
 = asyncio.run(run_async_code_551060f9())
logger.success(format_json())

"""
</CodeGroup>

## Real-World Use Cases

<Tabs>
  <Tab title="Customer Support">
"""
logger.info("## Real-World Use Cases")

messages = [
    {"role": "user", "content": "My subscription isn't working"},
    {"role": "assistant", "content": "I can help with that. What specific issue are you experiencing?"},
    {"role": "user", "content": "I can't access premium features even though I paid"}
]

client.add(messages,
    user_id="customer123",
    run_id="ticket-2024-001",
    version="v2"
)

"""
</Tab>
  <Tab title="Personal AI Assistant">
"""

preference_messages = [
    {"role": "user", "content": "I prefer morning workouts and vegetarian meals"},
    {"role": "assistant", "content": "Got it! I'll keep your fitness and dietary preferences in mind."}
]

client.add(preference_messages, user_id="user456", version="v2")

planning_messages = [
    {"role": "user", "content": "Help me plan tomorrow's schedule"},
    {"role": "assistant", "content": "Of course! I'll consider your morning workout preference."}
]

client.add(planning_messages,
    user_id="user456",
    run_id="daily-plan-2024-01-15",
    version="v2"
)

"""
</Tab>
  <Tab title="Educational Platform">
"""

profile_messages = [
    {"role": "user", "content": "I'm studying computer science and struggle with math"},
    {"role": "assistant", "content": "I'll tailor explanations to help with math concepts."}
]

client.add(profile_messages, user_id="student789", version="v2")

lesson_messages = [
    {"role": "user", "content": "Can you explain algorithms?"},
    {"role": "assistant", "content": "Sure! I'll explain algorithms with math-friendly examples."}
]

client.add(lesson_messages,
    user_id="student789",
    run_id="algorithms-lesson-1",
    version="v2"
)

"""
</Tab>
</Tabs>

## Best Practices

### ✅ Do
- **Organize by context scope**: Use `user_id` only for persistent data, add `run_id` for session-specific context
- **Keep messages focused** on the current interaction
- **Test with real interaction flows** to ensure context works as expected

### ❌ Don't
- Send duplicate messages or interaction history
- Forget to include `version="v2"` parameter
- Mix contextual and non-contextual approaches in the same application

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **Context not working** | Ensure you're using `version="v2"` and consistent `user_id` |
| **Wrong context retrieved** | Check if you need separate `run_id` values for different interaction topics |
| **Missing interaction history** | Verify all messages in the interaction thread use the same `user_id` and `run_id` |
| **Too much irrelevant context** | Use more specific `run_id` values to separate different interaction types |


<Snippet file="get-help.mdx" />
"""
logger.info("## Best Practices")

logger.info("\n\n[DONE]", bright=True)