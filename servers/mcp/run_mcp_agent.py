import asyncio

from jet.servers.mcp.mcp_agent import chat_session


if __name__ == "__main__":
    asyncio.run(chat_session())
