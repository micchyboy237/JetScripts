import asyncio
from jet.transformers.formatters import format_json
from jet.logger import CustomLogger
from mem0 import MemoryClient
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
title: Webhooks
description: 'Configure and manage webhooks to receive real-time notifications about memory events'
icon: "webhook"
iconType: "solid"
---

## Overview

Webhooks enable real-time notifications for memory events in your Mem0 project. Webhooks are configured at the project level, meaning each webhook is tied to a specific project and receives events solely from that project. You can configure webhooks to send HTTP POST requests to your specified URLs whenever memories are created, updated, or deleted.

## Managing Webhooks

### Create Webhook

Create a webhook for your project; it will receive events only from that project:
<CodeGroup>
"""
logger.info("## Overview")


os.environ["MEM0_API_KEY"] = "your-api-key"

client = MemoryClient()

webhook = client.create_webhook(
    url="https://your-app.com/webhook",
    name="Memory Logger",
    project_id="proj_123",
    event_types=["memory_add"]
)
logger.debug(webhook)

"""

"""

{ MemoryClient } = require('mem0ai')
client = new MemoryClient({ apiKey: 'your-api-key'})

async def run_async_code_67947d0d():
    webhook = await client.createWebhook({
    return webhook
webhook = asyncio.run(run_async_code_67947d0d())
logger.success(format_json(webhook))
    url: "https://your-app.com/webhook",
    name: "Memory Logger",
    projectId: "proj_123",
    eventTypes: ["memory_add"]
})
console.log(webhook)

"""

"""

{
  "webhook_id": "wh_123",
  "name": "Memory Logger",
  "url": "https://your-app.com/webhook",
  "event_types": ["memory_add"],
  "project": "default-project",
  "is_active": true,
  "created_at": "2025-02-18T22:59:56.804993-08:00",
  "updated_at": "2025-02-18T23:06:41.479361-08:00"
}

"""
</CodeGroup>

### Get Webhooks

Retrieve all webhooks for your project:

<CodeGroup>
"""
logger.info("### Get Webhooks")

webhooks = client.get_webhooks(project_id="proj_123")
logger.debug(webhooks)

"""

"""

async def run_async_code_9b02df66():
    async def run_async_code_93e8cb17():
        webhooks = await client.getWebhooks({projectId: "proj_123"})
        return webhooks
    webhooks = asyncio.run(run_async_code_93e8cb17())
    logger.success(format_json(webhooks))
    return webhooks
webhooks = asyncio.run(run_async_code_9b02df66())
logger.success(format_json(webhooks))
console.log(webhooks)

"""

"""

[
    {
        "webhook_id": "wh_123",
        "url": "https://mem0.ai",
        "name": "mem0",
        "owner": "john",
        "event_types": ["memory_add"],
        "project": "default-project",
        "is_active": true,
        "created_at": "2025-02-18T22:59:56.804993-08:00",
        "updated_at": "2025-02-18T23:06:41.479361-08:00"
    }
]

"""
</CodeGroup>

### Update Webhook

Update an existing webhook’s configuration by specifying its `webhook_id`:

<CodeGroup>
"""
logger.info("### Update Webhook")

updated_webhook = client.update_webhook(
    name="Updated Logger",
    url="https://your-app.com/new-webhook",
    event_types=["memory_update", "memory_add"],
    webhook_id="wh_123"
)
logger.debug(updated_webhook)

"""

"""

async def run_async_code_3c19518a():
    updatedWebhook = await client.updateWebhook({
    return updatedWebhook
updatedWebhook = asyncio.run(run_async_code_3c19518a())
logger.success(format_json(updatedWebhook))
    name: "Updated Logger",
    url: "https://your-app.com/new-webhook",
    eventTypes: ["memory_update", "memory_add"],
    webhookId: "wh_123"
})
console.log(updatedWebhook)

"""

"""

{
  "message": "Webhook updated successfully"
}

"""
</CodeGroup>

### Delete Webhook

Delete a webhook by providing its `webhook_id`:

<CodeGroup>
"""
logger.info("### Delete Webhook")

response = client.delete_webhook(webhook_id="wh_123")
logger.debug(response)

"""

"""

async def run_async_code_28941699():
    async def run_async_code_bd6bcd45():
        response = await client.deleteWebhook({webhookId: "wh_123"})
        return response
    response = asyncio.run(run_async_code_bd6bcd45())
    logger.success(format_json(response))
    return response
response = asyncio.run(run_async_code_28941699())
logger.success(format_json(response))
console.log(response)

"""

"""

{
  "message": "Webhook deleted successfully"
}

"""
</CodeGroup>

## Event Types

Mem0 supports the following event types for webhooks:

- `memory_add`: Triggered when a memory is added.
- `memory_update`: Triggered when an existing memory is updated.
- `memory_delete`: Triggered when a memory is deleted.

## Webhook Payload

When a memory event occurs, Mem0 sends an HTTP POST request to your webhook URL with the following payload:
"""
logger.info("## Event Types")

{
    "event_details": {
        "id": "a1b2c3d4-e5f6-4g7h-8i9j-k0l1m2n3o4p5",
            "data": {
            "memory": "Name is Alex"
            },
        "event": "ADD"
    }
}

"""
## Best Practices

1. **Implement Retry Logic**: Ensure your webhook endpoint can handle temporary failures by implementing retry logic.

2. **Verify Webhook Source**: Implement security measures to verify that webhook requests originate from Mem0.

3. **Process Events Asynchronously**: Process webhook events asynchronously to avoid timeouts and ensure reliable handling.

4. **Monitor Webhook Health**: Regularly review your webhook logs to ensure functionality and promptly address any delivery failures.

If you have any questions, please feel free to reach out to us using one of the following methods:

<Snippet file="get-help.mdx" />
"""
logger.info("## Best Practices")

logger.info("\n\n[DONE]", bright=True)