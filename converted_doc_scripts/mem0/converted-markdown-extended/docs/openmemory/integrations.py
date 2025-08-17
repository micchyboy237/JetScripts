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
title: MCP Client Integration Guide
icon: "plug"
iconType: "solid"
---

## Connecting an MCP Client

Once your OpenMemory server is running locally, you can connect any compatible MCP client to your personal memory stream. This enables a seamless memory layer integration for AI tools and agents.

Ensure the following environment variables are correctly set in your configuration files:

**In `/ui/.env`:**
"""
logger.info("## Connecting an MCP Client")

NEXT_PUBLIC_API_URL=http://localhost:8765
NEXT_PUBLIC_USER_ID=<user-id>

"""
**In `/api/.env`:**
"""

# OPENAI_API_KEY=sk-xxx
USER=<user-id>

"""
These values define where your MCP server is running and which user's memory is accessed.

### MCP Client Setup

Use the following one step command to configure OpenMemory Local MCP to a client. The general command format is as follows:
"""
logger.info("### MCP Client Setup")

npx @openmemory/install local http://localhost:8765/mcp/<client-name>/sse/<user-id> --client <client-name>

"""
Replace `<client-name>` with the desired client name and `<user-id>` with the value specified in your environment variables.

### Example Commands for Supported Clients

| Client      | Command |
|-------------|---------|
| Claude      | `npx install-mcp http://localhost:8765/mcp/claude/sse/<user-id> --client claude` |
| Cursor      | `npx install-mcp http://localhost:8765/mcp/cursor/sse/<user-id> --client cursor` |
| Cline       | `npx install-mcp http://localhost:8765/mcp/cline/sse/<user-id> --client cline` |
| RooCline    | `npx install-mcp http://localhost:8765/mcp/roocline/sse/<user-id> --client roocline` |
| Windsurf    | `npx install-mcp http://localhost:8765/mcp/windsurf/sse/<user-id> --client windsurf` |
| Witsy       | `npx install-mcp http://localhost:8765/mcp/witsy/sse/<user-id> --client witsy` |
| Enconvo     | `npx install-mcp http://localhost:8765/mcp/enconvo/sse/<user-id> --client enconvo` |
| Augment     | `npx install-mcp http://localhost:8765/mcp/augment/sse/<user-id> --client augment` |

### What This Does

Running one of the above commands registers the specified MCP client and connects it to your OpenMemory server. This enables the client to stream and store contextual memory for the provided user ID.

The connection status and memory activity can be monitored via the OpenMemory UI at [http://localhost:3000](http://localhost:3000).
"""
logger.info("### Example Commands for Supported Clients")

logger.info("\n\n[DONE]", bright=True)