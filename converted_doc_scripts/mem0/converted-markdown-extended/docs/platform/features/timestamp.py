from datetime import datetime, timedelta
from jet.logger import CustomLogger
from mem0 import MemoryClient
import MemoryClient from 'mem0ai'
import os
import shutil
import time


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
---
title: Memory Timestamps
description: 'Add timestamps to your memories to maintain chronological accuracy and historical context'
icon: "clock"
iconType: "solid"
---

## Overview

The Memory Timestamps feature allows you to specify when a memory was created, regardless of when it's actually added to the system. This powerful capability enables you to:

- Maintain accurate chronological ordering of memories
- Import historical data with proper timestamps
- Create memories that reflect when events actually occurred
- Build timelines with precise temporal information

By leveraging custom timestamps, you can ensure that your memory system maintains an accurate representation of when information was generated or events occurred.

## Benefits of Custom Timestamps

Custom timestamps offer several important benefits:

• **Historical Accuracy**: Preserve the exact timing of past events and information.

• **Data Migration**: Seamlessly migrate existing data while maintaining original timestamps.

• **Time-Sensitive Analysis**: Enable time-based analysis and pattern recognition across memories.

• **Consistent Chronology**: Maintain proper ordering of memories for coherent storytelling.

## Using Custom Timestamps

When adding new memories, you can specify a custom timestamp to indicate when the memory was created. This timestamp will be used instead of the current time.

### Adding Memories with Custom Timestamps

<CodeGroup>
"""
logger.info("## Overview")



os.environ["MEM0_API_KEY"] = "your-api-key"

client = MemoryClient()

current_time = datetime.now()

five_days_ago = current_time - timedelta(days=5)

unix_timestamp = int(five_days_ago.timestamp())

messages = [
    {"role": "user", "content": "I'm travelling to SF"}
]
client.add(messages, user_id="user1", timestamp=unix_timestamp)

"""

"""

client = new MemoryClient({ apiKey: 'your-api-key' })

currentTime = new Date()

fiveDaysAgo = new Date()
fiveDaysAgo.setDate(currentTime.getDate() - 5)

unixTimestamp = Math.floor(fiveDaysAgo.getTime() / 1000)

messages = [
    {"role": "user", "content": "I'm travelling to SF"}
]
client.add(messages, { user_id: "user1", timestamp: unixTimestamp })
    .then(response => console.log(response))
    .catch(error => console.error(error))

"""

"""

curl -X POST "https://api.mem0.ai/v1/memories/" \
     -H "Authorization: Token your-api-key" \
     -H "Content-Type: application/json" \
     -d '{
         "messages": [{"role": "user", "content": "I'm travelling to SF"}],
         "user_id": "user1",
         "timestamp": 1721577600
     }'

"""

"""

{
    "results": [
        {
            "id": "a1b2c3d4-e5f6-4g7h-8i9j-k0l1m2n3o4p5",
            "data": {"memory": "Travelling to SF"},
            "event": "ADD"
        }
    ]
}

"""
</CodeGroup>

### Timestamp Format

When specifying a custom timestamp, you should provide a Unix timestamp (seconds since epoch). This is an integer representing the number of seconds that have elapsed since January 1, 1970 (UTC).

For example, to create a memory with a timestamp of January 1, 2023:

<CodeGroup>
"""
logger.info("### Timestamp Format")

january_2023_timestamp = 1672531200  # Unix timestamp for 2023-01-01 00:00:00 UTC

messages = [
    {"role": "user", "content": "I'm travelling to SF"}
]
client.add(messages, user_id="user1", timestamp=january_2023_timestamp)

"""

"""

january2023Timestamp = 1672531200;  // Unix timestamp for 2023-01-01 00:00:00 UTC

messages = [
    {"role": "user", "content": "I'm travelling to SF"}
]
client.add(messages, { user_id: "user1", timestamp: january2023Timestamp })
    .then(response => console.log(response))
    .catch(error => console.error(error))

"""
</CodeGroup>

If you have any questions, please feel free to reach out to us using one of the following methods:

<Snippet file="get-help.mdx" />
"""
logger.info("If you have any questions, please feel free to reach out to us using one of the following methods:")

logger.info("\n\n[DONE]", bright=True)