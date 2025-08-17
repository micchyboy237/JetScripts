import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
import os
import shutil
import { Memory } from 'mem0ai/oss'


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
[Cloudflare Vectorize](https://developers.cloudflare.com/vectorize/) is a vector database offering from Cloudflare, allowing you to build AI-powered applications with vector embeddings.

### Usage

<CodeGroup>
"""
logger.info("### Usage")


config = {
  vectorStore: {
    provider: 'vectorize',
    config: {
      indexName: 'my-memory-index',
      accountId: 'your-cloudflare-account-id',
      apiKey: 'your-cloudflare-api-key',
      dimension: 1536, // Optional: defaults to 1536
    },
  },
}

memory = new Memory(config)
messages = [
    {"role": "user", "content": "I'm looking for a good book to read."},
    {"role": "assistant", "content": "Sure, what genre are you interested in?"},
    {"role": "user", "content": "I enjoy fantasy novels with strong world-building."},
    {"role": "assistant", "content": "Great! I'll keep that in mind for future recommendations."}
]
async def run_async_code_1302f5a1():
    await memory.add(messages, { userId: "bob", metadata: { interest: "books" } })
    return 
 = asyncio.run(run_async_code_1302f5a1())
logger.success(format_json())

"""
</CodeGroup>

### Config

Let's see the available parameters for the `vectorize` config:

<Tabs>
<Tab title="TypeScript">
| Parameter | Description | Default Value |
| --- | --- | --- |
| `indexName` | The name of the Vectorize index | `None` (Required) |
| `accountId` | Your Cloudflare account ID | `None` (Required) |
| `apiKey` | Your Cloudflare API token | `None` (Required) |
| `dimension` | Dimensions of the embedding model | `1536` |
</Tab>
</Tabs>
"""
logger.info("### Config")

logger.info("\n\n[DONE]", bright=True)