import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
from mem0 import AsyncMemoryClient
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
title: Advanced Memory Operations
description: 'Comprehensive guide to advanced memory operations and features'
icon: "gear"
iconType: "solid"
---

This guide covers advanced memory operations including complex filtering, batch operations, and detailed API usage. If you're just getting started, check out the [Quickstart](/platform/quickstart) first.

## Advanced Memory Creation

### Async Client (Python)

For asynchronous operations in Python, use the AsyncMemoryClient:
"""
logger.info("## Advanced Memory Creation")


os.environ["MEM0_API_KEY"] = "your-api-key"
client = AsyncMemoryClient()

async def main():
    messages = [
        {"role": "user", "content": "I'm travelling to SF"}
    ]
    async def run_async_code_b859dc45():
        async def run_async_code_327af25d():
            response = await client.add(messages, user_id="john")
            return response
        response = asyncio.run(run_async_code_327af25d())
        logger.success(format_json(response))
        return response
    response = asyncio.run(run_async_code_b859dc45())
    logger.success(format_json(response))
    logger.debug(response)

async def run_async_code_ba09313d():
    await main()
    return 
 = asyncio.run(run_async_code_ba09313d())
logger.success(format_json())

"""
### Detailed Memory Creation Examples

#### Long-term memory with full context

<CodeGroup>
"""
logger.info("### Detailed Memory Creation Examples")

messages = [
    {"role": "user", "content": "Hi, I'm Alex. I'm a vegetarian and I'm allergic to nuts."},
    {"role": "assistant", "content": "Hello Alex! I've noted that you're a vegetarian and have a nut allergy. I'll keep this in mind for any food-related recommendations or discussions."}
]

client.add(messages, user_id="alex", metadata={"food": "vegan"})

"""

"""

messages = [
    {"role": "user", "content": "Hi, I'm Alex. I'm a vegetarian and I'm allergic to nuts."},
    {"role": "assistant", "content": "Hello Alex! I've noted that you're a vegetarian and have a nut allergy. I'll keep this in mind for any food-related recommendations or discussions."}
]
client.add(messages, { user_id: "alex", metadata: { food: "vegan" } })
    .then(response => console.log(response))
    .catch(error => console.error(error))

"""

"""

curl -X POST "https://api.mem0.ai/v1/memories/" \
     -H "Authorization: Token your-api-key" \
     -H "Content-Type: application/json" \
     -d '{
         "messages": [
             {"role": "user", "content": "Hi, I'm Alex. I'm a vegetarian and I'm allergic to nuts."},
             {"role": "assistant", "content": "Hello Alex! I've noted that you're a vegetarian and have a nut allergy. I'll keep this in mind for any food-related recommendations or discussions."}
         ],
         "user_id": "alex",
         "metadata": {
             "food": "vegan"
         }
     }'

"""

"""

{
    "results": [
        {
            "memory": "Name is Alex",
            "event": "ADD"
        },
        {
            "memory": "Is a vegetarian",
            "event": "ADD"
        },
        {
            "memory": "Is allergic to nuts",
            "event": "ADD"
        }
    ]
}

"""
</CodeGroup>

<Note>
    When passing `user_id`, memories are primarily created based on user messages, but may be influenced by assistant messages for contextual understanding. For example, in a conversation about food preferences, both the user's stated preferences and their responses to the assistant's questions would form user memories. Similarly, when using `agent_id`, assistant messages are prioritized, but user messages might influence the agent's memories based on context.
    
    **Example:**
    ```
    User: My favorite cuisine is Italian
    Assistant: Nice! What about Indian cuisine?
    User: Don't like it much since I cannot eat spicy food
    
    Resulting user memories:
    memory1 - Likes Italian food
    memory2 - Doesn't like Indian food since cannot eat spicy 
    
    (memory2 comes from user's response about Indian cuisine)
    ```
</Note>

<Note>Metadata allows you to store structured information (location, timestamp, user state) with memories. Add it during creation to enable precise filtering and retrieval during searches.</Note>

#### Short-term memory for sessions

<CodeGroup>
"""
logger.info("#### Short-term memory for sessions")

messages = [
    {"role": "user", "content": "I'm planning a trip to Japan next month."},
    {"role": "assistant", "content": "That's exciting, Alex! A trip to Japan next month sounds wonderful. Would you like some recommendations for vegetarian-friendly restaurants in Japan?"},
    {"role": "user", "content": "Yes, please! Especially in Tokyo."},
    {"role": "assistant", "content": "Great! I'll remember that you're interested in vegetarian restaurants in Tokyo for your upcoming trip. I'll prepare a list for you in our next interaction."}
]

client.add(messages, user_id="alex", run_id="trip-planning-2024")

"""

"""

messages = [
    {"role": "user", "content": "I'm planning a trip to Japan next month."},
    {"role": "assistant", "content": "That's exciting, Alex! A trip to Japan next month sounds wonderful. Would you like some recommendations for vegetarian-friendly restaurants in Japan?"},
    {"role": "user", "content": "Yes, please! Especially in Tokyo."},
    {"role": "assistant", "content": "Great! I'll remember that you're interested in vegetarian restaurants in Tokyo for your upcoming trip. I'll prepare a list for you in our next interaction."}
]
client.add(messages, { user_id: "alex", run_id: "trip-planning-2024" })
    .then(response => console.log(response))
    .catch(error => console.error(error))

"""

"""

curl -X POST "https://api.mem0.ai/v1/memories/" \
     -H "Authorization: Token your-api-key" \
     -H "Content-Type: application/json" \
     -d '{
         "messages": [
             {"role": "user", "content": "I'm planning a trip to Japan next month."},
             {"role": "assistant", "content": "That's exciting, Alex! A trip to Japan next month sounds wonderful. Would you like some recommendations for vegetarian-friendly restaurants in Japan?"},
             {"role": "user", "content": "Yes, please! Especially in Tokyo."},
             {"role": "assistant", "content": "Great! I'll remember that you're interested in vegetarian restaurants in Tokyo for your upcoming trip. I'll prepare a list for you in our next interaction."}
         ],
         "user_id": "alex",
         "run_id": "trip-planning-2024"
     }'

"""

"""

{
  "results": [
    {
      "memory": "Planning a trip to Japan next month",
      "event": "ADD"
    },
    {
      "memory": "Interested in vegetarian restaurants in Tokyo",
      "event": "ADD"
    }
  ]
}

"""
</CodeGroup>

#### Agent memories

<CodeGroup>
"""
logger.info("#### Agent memories")

messages = [
    {"role": "system", "content": "You are an AI tutor with a personality. Give yourself a name for the user."},
    {"role": "assistant", "content": "Understood. I'm an AI tutor with a personality. My name is Alice."}
]

client.add(messages, agent_id="ai-tutor")

"""

"""

messages = [
    {"role": "system", "content": "You are an AI tutor with a personality. Give yourself a name for the user."},
    {"role": "assistant", "content": "Understood. I'm an AI tutor with a personality. My name is Alice."}
]
client.add(messages, { agent_id: "ai-tutor" })
    .then(response => console.log(response))
    .catch(error => console.error(error))

"""

"""

curl -X POST "https://api.mem0.ai/v1/memories/" \
     -H "Authorization: Token your-api-key" \
     -H "Content-Type: application/json" \
     -d '{
         "messages": [
             {"role": "system", "content": "You are an AI tutor with a personality. Give yourself a name for the user."},
             {"role": "assistant", "content": "Understood. I'm an AI tutor with a personality. My name is Alice."}
         ],
         "agent_id": "ai-tutor"
     }'

"""
</CodeGroup>

<Note>
    The `agent_id` retains memories exclusively based on messages generated by the assistant or those explicitly provided as input to the assistant. Messages outside these criteria are not stored as memory.
</Note>

#### Dual user and agent memories

When you provide both `user_id` and `agent_id`, Mem0 will store memories for both identifiers separately:
- Memories from messages with `"role": "user"` are automatically tagged with the provided `user_id`
- Memories from messages with `"role": "assistant"` are automatically tagged with the provided `agent_id`
- During retrieval, you can provide either `user_id` or `agent_id` to access the respective memories
- You can continuously enrich existing memory collections by adding new memories to the same `user_id` or `agent_id` in subsequent API calls, either together or separately, allowing for progressive memory building over time
- This dual-tagging approach enables personalized experiences for both users and AI agents in your application

<CodeGroup>
"""
logger.info("#### Dual user and agent memories")

messages = [
    {"role": "user", "content": "I'm travelling to San Francisco"},
    {"role": "assistant", "content": "That's great! I'm going to Dubai next month."},
]

client.add(messages=messages, user_id="user1", agent_id="agent1")

"""

"""

messages = [
    {"role": "user", "content": "I'm travelling to San Francisco"},
    {"role": "assistant", "content": "That's great! I'm going to Dubai next month."},
]

client.add(messages, { user_id: "user1", agent_id: "agent1" })
    .then(response => console.log(response))
    .catch(error => console.error(error))

"""

"""

curl -X POST "https://api.mem0.ai/v1/memories/" \
     -H "Authorization: Token your-api-key" \
     -H "Content-Type: application/json" \
     -d '{
         "messages": [
             {"role": "user", "content": "I'm travelling to San Francisco"},
             {"role": "assistant", "content": "That's great! I'm going to Dubai next month."},
         ],
         "user_id": "user1",
         "agent_id": "agent1"
     }'

"""

"""

{
    "results": [
        {
            "id": "c57abfa2-f0ac-48af-896a-21728dbcecee0",
            "data": {"memory": "Travelling to San Francisco"},
            "event": "ADD"
        },
        {
            "id": "0e8c003f-7db7-426a-9fdc-a46f9331a0c2",
            "data": {"memory": "Going to Dubai next month"},
            "event": "ADD"
        }
    ]
}

"""
</CodeGroup>

## Advanced Search Operations

### Search with Custom Filters

Our advanced search allows you to set custom search filters. You can filter by user_id, agent_id, app_id, run_id, created_at, updated_at, categories, and text. The filters support logical operators (AND, OR) and comparison operators (in, gte, lte, gt, lt, ne, contains, icontains, *). The wildcard character (*) matches everything for a specific field.

Here you need to define `version` as `v2` in the search method.

#### Example 1: Search using user_id and agent_id filters

<CodeGroup>
"""
logger.info("## Advanced Search Operations")

query = "What do you know about me?"
filters = {
   "OR":[
      {
         "user_id":"alex"
      },
      {
         "agent_id":{
            "in":[
               "travel-assistant",
               "customer-support"
            ]
         }
      }
   ]
}
client.search(query, version="v2", filters=filters)

"""

"""

query = "What do you know about me?"
filters = {
   "OR":[
      {
         "user_id":"alex"
      },
      {
         "agent_id":{
            "in":[
               "travel-assistant",
               "customer-support"
            ]
         }
      }
   ]
}
client.search(query, { version: "v2", filters })
    .then(results => console.log(results))
    .catch(error => console.error(error))

"""

"""

curl -X POST "https://api.mem0.ai/v1/memories/search/?version=v2" \
     -H "Authorization: Token your-api-key" \
     -H "Content-Type: application/json" \
     -d '{
         "query": "What do you know about me?",
         "filters": {
             "OR": [
                 {
                     "user_id": "alex"
                 },
                 {
                     "agent_id": {
                         "in": ["travel-assistant", "customer-support"]
                     }
                 }
             ]
         }
     }'

"""
</CodeGroup>

#### Example 2: Search using date filters

<CodeGroup>
"""
logger.info("#### Example 2: Search using date filters")

query = "What do you know about me?"
filters = {
    "AND": [
        {"created_at": {"gte": "2024-07-20", "lte": "2024-07-10"}},
        {"user_id": "alex"}
    ]
}
client.search(query, version="v2", filters=filters)

"""

"""

query = "What do you know about me?"
filters = {
  "AND": [
    {"created_at": {"gte": "2024-07-20", "lte": "2024-07-10"}},
    {"user_id": "alex"}
  ]
}

client.search(query, { version: "v2", filters })
  .then(results => console.log(results))
  .catch(error => console.error(error))

"""

"""

curl -X POST "https://api.mem0.ai/v1/memories/search/?version=v2" \
     -H "Authorization: Token your-api-key" \
     -H "Content-Type: application/json" \
     -d '{
         "query": "What do you know about me?",
         "filters": {
             "AND": [
                 {
                     "created_at": {
                         "gte": "2024-07-20",
                         "lte": "2024-07-10"
                     }
                 },
                 {
                     "user_id": "alex"
                 }
             ]
         }
     }'

"""
</CodeGroup>

#### Example 3: Search using metadata and categories

<CodeGroup>
"""
logger.info("#### Example 3: Search using metadata and categories")

query = "What do you know about me?"
filters = {
    "AND": [
        {"metadata": {"food": "vegan"}},
        {
         "categories":{
            "contains": "food_preferences"
         }
    }
    ]
}
client.search(query, version="v2", filters=filters)

"""

"""

query = "What do you know about me?"
filters = {
    "AND": [
        {"metadata": {"food": "vegan"}},
        {
            "categories": {
                "contains": "food_preferences"
            }
        }
    ]
}

client.search(query, { version: "v2", filters })
    .then(results => console.log(results))
    .catch(error => console.error(error))

"""

"""

curl -X POST "https://api.mem0.ai/v1/memories/search/?version=v2" \
     -H "Authorization: Token your-api-key" \
     -H "Content-Type: application/json" \
     -d '{
         "query": "What do you know about me?",
         "filters": {
             "AND": [
                 {
                     "metadata": {
                         "food": "vegan"
                     }
                 },
                 {
                     "categories": {
                         "contains": "food_preferences"
                     }
                 }
             ]
         }
     }'

"""
</CodeGroup>

#### Example 4: Search using NOT filters

<CodeGroup>
"""
logger.info("#### Example 4: Search using NOT filters")

query = "What do you know about me?"
filters = {
    "NOT": [
        {
            "categories": {
                "contains": "food_preferences"
            }
        }
    ]
}
client.search(query, version="v2", filters=filters)

"""

"""

query = "What do you know about me?"
filters = {
    "NOT": [
        {
            "categories": {
                "contains": "food_preferences"
            }
        }
    ]
}

client.search(query, { version: "v2", filters })
    .then(results => console.log(results))
    .catch(error => console.error(error))

"""

"""

curl -X POST "https://api.mem0.ai/v1/memories/search/?version=v2" \
     -H "Authorization: Token your-api-key" \
     -H "Content-Type: application/json" \
     -d '{
         "query": "What do you know about me?",
         "filters": {
            "NOT": [
                {
                    "categories": {
                        "contains": "food_preferences"
                    }
                }
            ]
        }
     }'

"""
</CodeGroup>

#### Example 5: Search using wildcard filters

<CodeGroup>
"""
logger.info("#### Example 5: Search using wildcard filters")

query = "What do you know about me?"
filters = {
    "AND": [
        {
            "user_id": "alex"
        },
        {
            "run_id": "*"  # Matches all run_ids
        }
    ]
}
client.search(query, version="v2", filters=filters)

"""

"""

query = "What do you know about me?"
filters = {
    "AND": [
        {
            "user_id": "alex"
        },
        {
            "run_id": "*"  // Matches all run_ids
        }
    ]
}

client.search(query, { version: "v2", filters })
    .then(results => console.log(results))
    .catch(error => console.error(error))

"""

"""

curl -X POST "https://api.mem0.ai/v1/memories/search/?version=v2" \
     -H "Authorization: Token your-api-key" \
     -H "Content-Type: application/json" \
     -d '{
         "query": "What do you know about me?",
         "filters": {
             "AND": [
                 {
                     "user_id": "alex"
                 },
                 {
                     "run_id": "*"
                 }
             ]
         }
     }'

"""
</CodeGroup>

## Advanced Retrieval Operations

### Get All Memories with Pagination

<Note> The `get_all` method supports two output formats: `v1.0` (default) and `v1.1`. To use the latest format, which provides more detailed information about each memory operation, set the `output_format` parameter to `v1.1`.</Note>

<Note> We're soon deprecating the default output format for get_all() method, which returned a list. Once the changes are live, paginated response will be the only supported format, with 100 memories per page by default. You can customize this using the `page` and `page_size` parameters.</Note>

#### Get all memories of a user

<CodeGroup>
"""
logger.info("## Advanced Retrieval Operations")

memories = client.get_all(user_id="alex", page=1, page_size=50)

"""

"""

client.getAll({ user_id: "alex", page: 1, page_size: 50 })
    .then(memories => console.log(memories))
    .catch(error => console.error(error))

"""

"""

curl -X GET "https://api.mem0.ai/v1/memories/?user_id=alex&page=1&page_size=50" \
     -H "Authorization: Token your-api-key"

"""

"""

{
    "count": 204,
    "next": "https://api.mem0.ai/v1/memories/?user_id=alex&output_format=v1.1&page=2&page_size=50",
    "previous": null,
    "results": [
        {
            "id":"f38b689d-6b24-45b7-bced-17fbb4d8bac7",
            "memory":"是素食主义者，对坚果过敏。",
            "agent_id":"travel-assistant",
            "hash":"62bc074f56d1f909f1b4c2b639f56f6a",
            "metadata":null,
            "immutable": false,
            "expiration_date": null,
            "created_at":"2024-07-25T23:57:00.108347-07:00",
            "updated_at":"2024-07-25T23:57:00.108367-07:00",
            "categories":null
        }
    ]
}

"""
</CodeGroup>

#### Get all memories by categories

You can filter memories by their categories when using get_all:

<CodeGroup>
"""
logger.info("#### Get all memories by categories")

memories = client.get_all(user_id="alex", categories=["likes"])

memories = client.get_all(user_id="alex", categories=["likes", "food_preferences"])

memories = client.get_all(user_id="alex", categories=["likes"], page=1, page_size=50)

memories = client.get_all(user_id="alex", keywords="to play", page=1, page_size=50)

"""

"""

client.getAll({ user_id: "alex", categories: ["likes"] })
    .then(memories => console.log(memories))
    .catch(error => console.error(error))

client.getAll({ user_id: "alex", categories: ["likes", "food_preferences"] })
    .then(memories => console.log(memories))
    .catch(error => console.error(error))

client.getAll({ user_id: "alex", categories: ["likes"], page: 1, page_size: 50 })
    .then(memories => console.log(memories))
    .catch(error => console.error(error))

client.getAll({ user_id: "alex", keywords: "to play", page: 1, page_size: 50 })
    .then(memories => console.log(memories))
    .catch(error => console.error(error))

"""

"""

curl -X GET "https://api.mem0.ai/v1/memories/?user_id=alex&categories=likes" \
     -H "Authorization: Token your-api-key"

curl -X GET "https://api.mem0.ai/v1/memories/?user_id=alex&categories=likes,food_preferences" \
     -H "Authorization: Token your-api-key"

curl -X GET "https://api.mem0.ai/v1/memories/?user_id=alex&categories=likes&page=1&page_size=50" \
     -H "Authorization: Token your-api-key"

curl -X GET "https://api.mem0.ai/v1/memories/?user_id=alex&keywords=to play&page=1&page_size=50" \
     -H "Authorization: Token your-api-key"

"""
</CodeGroup>

#### Get all memories using custom filters

Our advanced retrieval allows you to set custom filters when fetching memories. You can filter by user_id, agent_id, app_id, run_id, created_at, updated_at, categories, and keywords. The filters support logical operators (AND, OR) and comparison operators (in, gte, lte, gt, lt, ne, contains, icontains, *). The wildcard character (*) matches everything for a specific field.

Here you need to define `version` as `v2` in the get_all method.

<CodeGroup>
"""
logger.info("#### Get all memories using custom filters")

filters = {
   "AND":[
      {
         "user_id":"alex"
      },
      {
         "created_at":{
            "gte":"2024-07-01",
            "lte":"2024-07-31"
         }
      },
      {
         "categories":{
            "contains": "food_preferences"
         }
      }
   ]
}

client.get_all(version="v2", filters=filters)

client.get_all(version="v2", filters=filters, page=1, page_size=50)

"""

"""

filters = {
   "AND":[
      {
         "user_id":"alex"
      },
      {
         "created_at":{
            "gte":"2024-07-01",
            "lte":"2024-07-31"
         }
      },
      {
         "categories":{
            "contains": "food_preferences"
         }
      }
   ]
}

client.getAll({ version: "v2", filters })
    .then(memories => console.log(memories))
    .catch(error => console.error(error))

client.getAll({ version: "v2", filters, page: 1, page_size: 50 })
    .then(memories => console.log(memories))
    .catch(error => console.error(error))

"""

"""

curl -X GET "https://api.mem0.ai/v1/memories/?version=v2" \
     -H "Authorization: Token your-api-key" \
     -H "Content-Type: application/json" \
     -d '{
         "filters": {
             "AND": [
                {"user_id":"alex"},
                {"created_at":{
                    "gte":"2024-07-01",
                    "lte":"2024-07-31"
                }},
                {"categories":{
                    "contains": "food_preferences"
                }}
             ]
         }
     }'

curl -X GET "https://api.mem0.ai/v1/memories/?version=v2&page=1&page_size=50" \
     -H "Authorization: Token your-api-key" \
     -H "Content-Type: application/json" \
     -d '{
         "filters": {
             "AND": [
                {"user_id":"alex"},
                {"created_at":{
                    "gte":"2024-07-01",
                    "lte":"2024-07-31"
                }},
                {"categories":{
                    "contains": "food_preferences"
                }}
             ]
         }
     }'

"""
</CodeGroup>

## Memory Management Operations

### Memory History

Get history of how a memory has changed over time.

<CodeGroup>
"""
logger.info("## Memory Management Operations")

messages = [{"role": "user", "content": "I recently tried chicken and I loved it. I'm thinking of trying more non-vegetarian dishes.."}]
client.add(messages, user_id="alex")

messages.append({'role': 'user', 'content': 'I turned vegetarian now.'})
client.add(messages, user_id="alex")

memory_id = "<memory-id-here>"
history = client.history(memory_id)

"""

"""

messages = [{ role: "user", content: "I recently tried chicken and I loved it. I'm thinking of trying more non-vegetarian dishes.." }]
client.add(messages, { user_id: "alex" })
    .then(result => {
        messages.push({ role: 'user', content: 'I turned vegetarian now.' })
        return client.add(messages, { user_id: "alex" })
    })
    .then(result => {
        memoryId = result.id; // Assuming the API returns the memory ID
        return client.history(memoryId)
    })
    .then(history => console.log(history))
    .catch(error => console.error(error))

"""

"""

curl -X POST "https://api.mem0.ai/v1/memories/" \
     -H "Authorization: Token your-api-key" \
     -H "Content-Type: application/json" \
     -d '{
         "messages": [{"role": "user", "content": "I recently tried chicken and I loved it. I'm thinking of trying more non-vegetarian dishes.."}],
         "user_id": "alex"
     }'

curl -X POST "https://api.mem0.ai/v1/memories/" \
     -H "Authorization: Token your-api-key" \
     -H "Content-Type: application/json" \
     -d '{
         "messages": [
             {"role": "user", "content": "I recently tried chicken and I loved it. I'm thinking of trying more non-vegetarian dishes.."},
             {"role": "user", "content": "I turned vegetarian now."}
         ],
         "user_id": "alex"
     }'

curl -X GET "https://api.mem0.ai/v1/memories/<memory-id-here>/history/" \
     -H "Authorization: Token your-api-key"

"""

"""

[
   {
      "id":"d6306e85-eaa6-400c-8c2f-ab994a8c4d09",
      "memory_id":"b163df0e-ebc8-4098-95df-3f70a733e198",
      "input":[
         {
            "role":"user",
            "content":"I recently tried chicken and I loved it. I'm thinking of trying more non-vegetarian dishes.."
         },
         {
            "role":"user",
            "content":"I turned vegetarian now."
         }
      ],
      "old_memory":"None",
      "new_memory":"Turned vegetarian.",
      "user_id":"alex",
      "event":"ADD",
      "metadata":"None",
      "created_at":"2024-07-26T01:02:41.737310-07:00",
      "updated_at":"2024-07-26T01:02:41.726073-07:00"
   }
]

"""
</CodeGroup>

### Update Memory

Update a memory with new data. You can update the memory's text, metadata, or both.

<CodeGroup>
"""
logger.info("### Update Memory")

client.update(
    memory_id="<memory-id-here>",
    text="I am now a vegetarian.",
    metadata={"diet": "vegetarian"}
)

"""

"""

client.update("memory-id-here", { text: "I am now a vegetarian.", metadata: { diet: "vegetarian" } })
    .then(result => console.log(result))
    .catch(error => console.error(error))

"""

"""

curl -X PUT "https://api.mem0.ai/v1/memories/<memory-id-here>" \
        -H "Authorization: Token your-api-key" \
        -H "Content-Type: application/json" \
        -d '{
            "message": "I recently tried chicken and I loved it. I'm thinking of trying more non-vegetarian dishes.."
        }'

"""

"""

{
   "id":"c190ab1a-a2f1-4f6f-914a-495e9a16b76e",
   "memory":"I recently tried chicken and I loved it. I'm thinking of trying more non-vegetarian dishes..",
   "agent_id":"travel-assistant",
   "hash":"af1161983e03667063d1abb60e6d5c06",
   "metadata":"None",
   "created_at":"2024-07-30T22:46:40.455758-07:00",
   "updated_at":"2024-07-30T22:48:35.257828-07:00"
}

"""
</CodeGroup>

## Batch Operations

### Batch Update Memories

Update multiple memories in a single API call. You can update up to 1000 memories at once.

<CodeGroup>
"""
logger.info("## Batch Operations")

update_memories = [
    {
        "memory_id": "285ed74b-6e05-4043-b16b-3abd5b533496",
        "text": "Watches football"
    },
    {
        "memory_id": "2c9bd859-d1b7-4d33-a6b8-94e0147c4f07",
        "text": "Loves to travel"
    }
]

response = client.batch_update(update_memories)
logger.debug(response)

"""

"""

updateMemories = [
    {
        "memory_id": "285ed74b-6e05-4043-b16b-3abd5b533496",
        text: "Watches football"
    },
    {
        "memory_id": "2c9bd859-d1b7-4d33-a6b8-94e0147c4f07",
        text: "Loves to travel"
    }
]

client.batchUpdate(updateMemories)
    .then(response => console.log('Batch update response:', response))
    .catch(error => console.error(error))

"""

"""

curl -X PUT "https://api.mem0.ai/v1/memories/batch/" \
     -H "Authorization: Token your-api-key" \
     -H "Content-Type: application/json" \
     -d '{
         "memories": [
             {
                 "memory_id": "285ed74b-6e05-4043-b16b-3abd5b533496",
                 "text": "Watches football"
             },
             {
                 "memory_id": "2c9bd859-d1b7-4d33-a6b8-94e0147c4f07",
                 "text": "Loves to travel"
             }
         ]
     }'

"""

"""

{
    "message": "Successfully updated 2 memories"
}

"""
</CodeGroup>

### Batch Delete Memories

Delete multiple memories in a single API call. You can delete up to 1000 memories at once.

<CodeGroup>
"""
logger.info("### Batch Delete Memories")

delete_memories = [
    {"memory_id": "285ed74b-6e05-4043-b16b-3abd5b533496"},
    {"memory_id": "2c9bd859-d1b7-4d33-a6b8-94e0147c4f07"}
]

response = client.batch_delete(delete_memories)
logger.debug(response)

"""

"""

deleteMemories = [
    {"memory_id": "285ed74b-6e05-4043-b16b-3abd5b533496"},
    {"memory_id": "2c9bd859-d1b7-4d33-a6b8-94e0147c4f07"}
]

client.batchDelete(deleteMemories)
    .then(response => console.log('Batch delete response:', response))
    .catch(error => console.error(error))

"""

"""

curl -X DELETE "https://api.mem0.ai/v1/memories/batch/" \
     -H "Authorization: Token your-api-key" \
     -H "Content-Type: application/json" \
     -d '{
         "memory_ids": [
             {"memory_id": "285ed74b-6e05-4043-b16b-3abd5b533496"},
             {"memory_id": "2c9bd859-d1b7-4d33-a6b8-94e0147c4f07"}
         ]
     }'

"""

"""

{
    "message": "Successfully deleted 2 memories"
}

"""
</CodeGroup>

## Entity Management

### Get All Users

Get all users, agents, and runs which have memories associated with them.

<CodeGroup>
"""
logger.info("## Entity Management")

client.users()

"""

"""

client.users()
    .then(users => console.log(users))
    .catch(error => console.error(error))

"""

"""

curl -X GET "https://api.mem0.ai/v1/entities/" \
     -H "Authorization: Token your-api-key"

"""

"""

[
    {
        "id": "1",
        "name": "user123",
        "created_at": "2024-07-17T16:47:23.899900-07:00",
        "updated_at": "2024-07-17T16:47:23.899918-07:00",
        "total_memories": 5,
        "owner": "alex",
        "metadata": {"foo": "bar"},
        "type": "user"
    },
    {
        "id": "2",
        "name": "travel-agent",
        "created_at": "2024-07-01T17:59:08.187250-07:00",
        "updated_at": "2024-07-01T17:59:08.187266-07:00",
        "total_memories": 10,
        "owner": "alex",
        "metadata": {"agent_id": "123"},
        "type": "agent"
    }
]

"""
</CodeGroup>

### Delete Operations

Delete specific memory:

<CodeGroup>
"""
logger.info("### Delete Operations")

client.delete(memory_id)

"""

"""

client.delete("memory-id-here")
    .then(result => console.log(result))
    .catch(error => console.error(error))

"""

"""

curl -X DELETE "https://api.mem0.ai/v1/memories/memory-id-here" \
     -H "Authorization: Token your-api-key"

"""
</CodeGroup>

Delete all memories of a user:

<CodeGroup>
"""
logger.info("Delete all memories of a user:")

client.delete_all(user_id="alex")

"""

"""

client.deleteAll({ user_id: "alex" })
    .then(result => console.log(result))
    .catch(error => console.error(error))

"""

"""

curl -X DELETE "https://api.mem0.ai/v1/memories/?user_id=alex" \
     -H "Authorization: Token your-api-key"

"""
</CodeGroup>

Delete specific user or agent:

<CodeGroup>
"""
logger.info("Delete specific user or agent:")

client.delete_users(user_id="alex")

"""

"""

client.delete_users({ user_id: "alex" })
    .then(result => console.log(result))
    .catch(error => console.error(error))

"""

"""

curl -X DELETE "https://api.mem0.ai/v2/entities/user/alex" \
     -H "Authorization: Token your-api-key"

"""
</CodeGroup>

### Reset Client

<CodeGroup>
"""
logger.info("### Reset Client")

client.reset()

"""

"""

{'message': 'Client reset successful. All users and memories deleted.'}

"""
</CodeGroup>

### Natural Language Delete

You can also delete memories using natural language commands:

<CodeGroup>
"""
logger.info("### Natural Language Delete")

messages = [
    {"role": "user", "content": "Delete all of my food preferences"}
]
client.add(messages, user_id="alex")

"""

"""

messages = [
    {"role": "user", "content": "Delete all of my food preferences"}
]
client.add(messages, { user_id: "alex" })
    .then(result => console.log(result))
    .catch(error => console.error(error))

"""

"""

curl -X POST "https://api.mem0.ai/v1/memories/" \
     -H "Authorization: Token your-api-key" \
     -H "Content-Type: application/json" \
     -d '{
         "messages": [{"role": "user", "content": "Delete all of my food preferences"}],
         "user_id": "alex"
     }'

"""
</CodeGroup>

## Monitor Memory Operations

You can monitor memory operations on the platform dashboard:

![Mem0 Platform Activity](/images/platform/activity.png)

For more detailed information, see our [API Reference](/api-reference) or explore specific features in the [Platform Features](/platform/features/platform-overview) section.
"""
logger.info("## Monitor Memory Operations")

logger.info("\n\n[DONE]", bright=True)