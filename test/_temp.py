from jet.logger import logger
from jet.servers.mcp.server.utils import parse_tool_requests
from jet.transformers.formatters import format_json


text = """
{
  "tool": "navigate_to_url",
  "arguments": {
    "url": "https://www.iana.org"
  }
}
{"tool": "summarize_text", "arguments": {"text": "<html><head><title>IANA</title></head><body><h1>International Assigned Numbers Authority</h1><p>IANA is the organization responsible for the allocation of numbers and codes used in the Internet. It manages the global system of domain names, internet protocols, and other technical standards.</p></body></html>", "max_words": 100}}
"""

result = parse_tool_requests(text, logger)
logger.success(format_json(result))
