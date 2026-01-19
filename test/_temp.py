# playwright_mcp_client.py
import asyncio
import json
from typing import Any, Dict, Optional, AsyncGenerator
import httpx
from rich.console import Console
from rich.logging import RichHandler
import logging
from jet.logger import logger as jlogger

jlogger.info("START")

# Very verbose logging for debugging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[RichHandler(rich_tracebacks=True, markup=True, show_time=True)]
)
logger = logging.getLogger("mcp_client")
console = Console()

class PlaywrightMCPClient:
    """JSON-RPC client for Playwright MCP with session-id support"""

    def __init__(self, base_url: str = "http://localhost:8931/mcp"):
        self.base_url = base_url.rstrip("/")
        # Force connection pooling & long keep-alive
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(10.0, connect=10.0, read=300.0, write=60.0),
            limits=httpx.Limits(
                max_connections=5,
                max_keepalive_connections=5,
                keepalive_expiry=300.0
            ),
            http2=False,
            follow_redirects=True
        )
        self.request_id = 0
        self.initialized = False
        self.mcp_session_id: Optional[str] = None

    async def _call(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        is_notification: bool = False
    ) -> Optional[Dict[str, Any]]:
        self.request_id += 1

        payload: Dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
        }

        if not is_notification:
            payload["id"] = self.request_id

        if params is not None:
            payload["params"] = params

        logger.debug("Sending payload:\n%s", json.dumps(payload, indent=2))

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream;q=0.9",
            "Connection": "keep-alive",
        }

        # --- Critical: send mcp-session-id if available ---
        if self.mcp_session_id:
            headers["mcp-session-id"] = self.mcp_session_id
            logger.debug("Using mcp-session-id: %s", self.mcp_session_id)

        logger.debug("Request headers: %s", headers)

        try:
            response = await self.client.post(
                self.base_url,
                json=payload,
                headers=headers
            )

            logger.info("Response status: %d  |  Content-Type: %s",
                        response.status_code,
                        response.headers.get("content-type", "unknown"))

            # If we haven't stored the session id, check for it
            session_id = response.headers.get("mcp-session-id")
            if not self.mcp_session_id and session_id:
                self.mcp_session_id = session_id
                logger.info("Received new mcp-session-id: %s", session_id)

            # Handle notification requests (no "id" field)
            if payload.get("id") is None:
                if response.status_code not in (200, 202, 204):
                    raise RuntimeError(f"Notification failed: {response.status_code} - {await response.aread()}")
                return None  # critical: do NOT try to parse SSE/body for notifications

            response.raise_for_status()

            # ─── Real-time SSE processing with immediate logging ─────────────────────
            final_result = None
            async for event in self._parse_sse(response):
                if "data" in event:
                    data_str = event["data"].strip()
                    if not data_str:
                        continue

                    logger.debug("[SSE chunk received] %s", data_str)

                    try:
                        parsed = json.loads(data_str)
                        logger.debug("[SSE parsed] %s", json.dumps(parsed, indent=2))

                        # Handle both "result" and "error" at top level
                        if "error" in parsed:
                            console.print("[bold red]→ Chunk ERROR:[/bold red]", parsed["error"])
                            raise RuntimeError(f"MCP error: {parsed['error']}")
                        elif "result" in parsed:
                            console.print("[dim bright_white]→ Chunk result:[/dim bright_white]", parsed["result"])
                            final_result = parsed["result"]
                        else:
                            # some servers send content in other fields
                            console.print("[dim yellow]→ Other SSE chunk:[/dim yellow]", parsed)
                            final_result = parsed

                        if "method" in parsed and parsed["method"] == "notifications":
                            console.print("[yellow]→ Server notification:[/yellow]", parsed)

                    except json.JSONDecodeError as e:
                        console.print("[dim yellow]→ Non-JSON chunk:[/dim yellow]", data_str)
                        logger.warning("Invalid JSON in SSE data: %s → %s", data_str, e)

            if final_result is None:
                logger.error("No valid 'result' received in any SSE chunk")
                raise RuntimeError("No valid 'result' received in any SSE chunk")

            return final_result

        except httpx.HTTPStatusError as e:
            try:
                error_body = await e.response.aread()
                error_text = error_body.decode(errors='replace')
            except Exception:
                error_text = "<error reading error body>"
            logger.error("HTTP %d: %s", e.response.status_code, error_text)
            raise
        except Exception as exc:
            logger.exception("Unexpected error in MCP call: %s", exc)
            raise

    async def _parse_sse(self, response: httpx.Response) -> AsyncGenerator[Dict[str, str], None]:
        buffer = ""
        async for chunk in response.aiter_text():
            buffer += chunk
            while "\n\n" in buffer:
                event_str, buffer = buffer.split("\n\n", 1)
                event_data = {}
                lines = event_str.splitlines()
                for line in lines:
                    line = line.strip()
                    if line.startswith("event:"):
                        event_data["event"] = line[6:].strip()
                    elif line.startswith("data:"):
                        data = line[5:].strip()
                        if "data" in event_data:
                            event_data["data"] += "\n" + data
                        else:
                            event_data["data"] = data
                if event_data:
                    yield event_data

        if buffer.strip():
            yield {"data": buffer.strip()}

    async def initialize(self) -> Optional[str]:
        if self.initialized:
            logger.info("Already initialized")
            return self.mcp_session_id

        init_params = {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {},
                "resources": {},
                "prompts": {},
                "sampling": {}
            },
            "clientInfo": {
                "name": "custom-python-mcp-client",
                "version": "0.1.0"
            }
        }

        logger.info("Performing MCP initialize...")

        try:
            # We temporarily override response handling to capture session id
            response = await self.client.post(
                self.base_url,
                json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": init_params},
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream;q=0.9",
                    "Connection": "keep-alive"
                }
            )

            logger.info("Response status: %d  |  Content-Type: %s", response.status_code, response.headers.get("content-type", "unknown"))
            response.raise_for_status()

            session_id = response.headers.get("mcp-session-id")
            if session_id:
                self.mcp_session_id = session_id
                logger.info("Received mcp-session-id: %s", session_id)
            else:
                logger.warning("Server did NOT return mcp-session-id header")

            # Now process the body normally (reuse existing parsing logic)
            final_result = None
            async for event in self._parse_sse(response):
                if "data" in event:
                    data_str = event["data"].strip()
                    if not data_str:
                        continue
                    logger.debug("[INIT chunk] %s", data_str)
                    try:
                        parsed = json.loads(data_str)
                        if "result" in parsed:
                            console.print("[dim bright_white]→ Init result:[/dim bright_white]", parsed["result"])
                            final_result = parsed["result"]
                    except json.JSONDecodeError:
                        logger.warning("Non-JSON chunk during init: %s", data_str)
                        pass

            if final_result:
                console.print("[bold green]Initialize success:[/bold green]", final_result)
            else:
                logger.error("Initialize did not return valid result")
                raise RuntimeError("Initialize did not return valid result")

            # Try to send notification (many implementations ignore or accept it)
            try:
                await self._call("initialized", None, is_notification=True)
                logger.info("initialized notification sent (no params)")
            except Exception as e:
                logger.warning("Could not send initialized notification: %s", e)

            self.initialized = True

            return self.mcp_session_id

        except httpx.HTTPError as e:
            logger.error("HTTP error during initialize: %s", e)
            raise
        except Exception as exc:
            logger.exception("Failed to initialize MCP session: %s", exc)
            raise

    async def browser_evaluate(self, expr_or_func: str, argument_key: str = "expression") -> Any:
        await self.initialize()

        logger.info("Calling browser_evaluate → %s", expr_or_func[:100] + "..." if len(expr_or_func) > 100 else expr_or_func)

        tool_params = {
            "name": "browser_evaluate",
            "arguments": {argument_key: expr_or_func}
        }
        # If "browser_evaluate" fails, you may try:
        # tool_params["name"] = "evaluate"
        # tool_params["name"] = "browser.evaluate"

        try:
            # Updated method and parameters per spec
            result = await self._call("tools/call", tool_params)
        except RuntimeError:
            # Optionally fallback to alternate tool names if needed; not automatic
            raise

        # For compat, if the result is a dict and contains 'success' == False
        if isinstance(result, dict) and result.get("success") is False:
            raise RuntimeError(f"Evaluation failed: {result.get('error', 'unknown error')}")

        # Defensive: return actual result if present, else result itself
        if isinstance(result, dict) and "result" in result:
            return result["result"]
        return result

    async def close(self):
        logger.info("Closing client connection pool...")
        await self.client.aclose()


# ─── Example usage ──────────────────────────────────────────────────────────────
async def main():
    client = PlaywrightMCPClient("http://localhost:8931/mcp")

    try:
        # Add navigation first (try these names one by one until one works)
        # Standard/expected name
        await client._call(
            "tools/call",
            {
                "name": "browser_navigate",
                "arguments": {"url": "https://example.com"}  # ← change to your real starting URL
            }
        )
        # or, if not supported:
        # await client._call("tools/call", {"name": "goto", "arguments": {"url": "https://example.com"}})
        # await client._call("tools/call", {"name": "page_goto", "arguments": {"url": "https://example.com"}})
        # await client._call("tools/call", {"name": "navigate", "arguments": {"url": "https://example.com"}})

        # Simple title - use arrow function wrapper
        title = await client.browser_evaluate("() => document.title", argument_key="function")
        console.print(f"\n[bold cyan]Final Page title:[/bold cyan] {title}\n")

        # Complex extraction - already good, just add navigation before it
        info = await client.browser_evaluate(
            "() => ({\n  title: document.title,\n  url: window.location.href,\n  links_count: document.querySelectorAll('a[href]').length\n})",
            argument_key="function"
        )

        console.print("[bold green]Final Page info:[/bold green]")
        console.print_json(data=info)

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())