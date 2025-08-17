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
title: 'Search Memories (v2)'
openapi: post /v2/memories/search/
---

The v2 search API is powerful and flexible, allowing for more precise memory retrieval. It supports complex logical operations (AND, OR, NOT) and comparison operators for advanced filtering capabilities. The comparison operators include:
- `in`: Matches any of the values specified
- `gte`: Greater than or equal to
- `lte`: Less than or equal to
- `gt`: Greater than
- `lt`: Less than
- `ne`: Not equal to
- `icontains`: Case-insensitive containment check
- `*`: Wildcard character that matches everything

  <CodeGroup>
  ```python Code
  related_memories = m.search(
      query="What are Alice's hobbies?",
      version="v2",
      filters={
          "OR": [
              {
                "user_id": "alice"
              },
              {
                "agent_id": {"in": ["travel-agent", "sports-agent"]}
              }
          ]
      },
  )
  ```

  ```json Output
  {
    "memories": [
      {
        "id": "ea925981-272f-40dd-b576-be64e4871429",
        "memory": "Likes to play cricket and plays cricket on weekends.",
        "metadata": {
          "category": "hobbies"
        },
        "score": 0.32116443111457704,
        "created_at": "2024-07-26T10:29:36.630547-07:00",
        "updated_at": null,
        "user_id": "alice",
        "agent_id": "sports-agent"
      }
    ],
  }
  ```
  </CodeGroup>

  <CodeGroup>
  ```python Wildcard Example
  # Using wildcard to match all run_ids for a specific user
  all_memories = m.search(
      query="What are Alice's hobbies?",
      version="v2",
      filters={
          "AND": [
              {
                  "user_id": "alice"
              },
              {
                  "run_id": "*"
              }
          ]
      },
  )
  ```
  </CodeGroup>
"""
logger.info("# Using wildcard to match all run_ids for a specific user")

logger.info("\n\n[DONE]", bright=True)