import json
import requests
import logging
from rich.console import Console
from rich.panel import Panel

# Configure rich logging
console = Console()

# Set up logging with rich formatting
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

# Create a custom rich logger
logger = logging.getLogger("llama_stream_client")

# Define the URL and payload
url = "http://shawn-pc.local:8080/v1/chat/completions"

payload = {
    "messages": [
        {
            "role": "user",
            "content": "Translate this Japanese text to english: \"おい、そんな一気に冷たいものを食べると腹を壊す\""
        }
    ],
    "stream": True,
    "cache_prompt": True,
    "reasoning_format": "none",
    "samplers": "edkypmxt",
    "temperature": 0.8,
    "dynatemp_range": 0,
    "dynatemp_exponent": 1,
    "top_k": 40,
    "top_p": 0.95,
    "min_p": 0.05,
    "typical_p": 1,
    "xtc_probability": 0,
    "xtc_threshold": 0.1,
    "repeat_last_n": 64,
    "repeat_penalty": 1,
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "dry_multiplier": 0,
    "dry_base": 1.75,
    "dry_allowed_length": 2,
    "dry_penalty_last_n": -1,
    "max_tokens": -1,
    "timings_per_token": False
}

headers = {
    "Content-Type": "application/json",
    "Accept": "*/*",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36 OPR/124.0.0.0",
    "Origin": "http://shawn-pc.local:8080",
    "Referer": "http://shawn-pc.local:8080/",
    "Host": "shawn-pc.local:8080"
}

# -------------------------------
# Rich Logging Setup (updated)
# -------------------------------
def log_info(msg: str):
    logger.info(msg)

def log_warning(msg: str):
    logger.warning(msg)

def log_error(msg: str):
    logger.error(msg)

def log_debug(msg: str):
    logger.debug(msg)

def log_header(title: str):
    panel = Panel(
        f"[bold magenta]{title}[/bold magenta]",
        border_style="bold blue",
        style="bold"
    )
    console.print(panel)

def log_stream_content(content: str):
    """Print only the actual generated text as it streams – no JSON noise"""
    if content and content.strip():
        console.print(content, end="", style="green")  # flush immediately

def log_final_summary(usage: dict | None, timings: dict | None = None):
    """Show a clean summary panel once the stream ends"""
    lines = ["[bold cyan]Stream Complete[/bold cyan]"]
    if usage:
        lines.append(f"  Prompt tokens : {usage.get('prompt_tokens', 'N/A')}")
        lines.append(f"  Completion    : {usage.get('completion_tokens', 'N/A')}")
        lines.append(f"  Total tokens  : {usage.get('total_tokens', 'N/A')}")
    if timings:
        lines.append("")
        lines.append("[dim]Timings[/dim]")
        lines.append(f"  Prompt ms       : {timings.get('prompt_ms', 'N/A')}")
        lines.append(f"  Predicted ms    : {timings.get('predicted_ms', 'N/A')}")
    panel = Panel("\n".join(lines), border_style="bold green", padding=(1, 2))
    console.print(panel)

# -------------------------------
# Main Execution
# -------------------------------
def main():
    log_header("🚀 Starting Llama.cpp Stream Request")

    try:
        # Make the POST request with streaming
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            stream=True,
            timeout=10
        )

        # Status check
        if response.status_code == 200:
            log_info("✅ Success: Connection established to Llama.cpp server")
        else:
            log_error(f"❌ HTTP {response.status_code} error: {response.text}")
            return

        # Log the response headers (optional)
        log_info("Headers received:")
        for k, v in response.headers.items():
            if k.lower() in ["content-type", "content-length", "transfer-encoding"]:
                log_info(f"  {k}: {v}")

        # ---- Updated streaming loop starts here ----
        log_info("Streaming response started...")

        full_content = ""
        final_usage = None
        final_timings = None

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue

            line = line.strip()

            if not line:
                continue

            # llama.cpp sends [DONE] as plain text, not inside data:
            if line == "[DONE]":
                break

            if line.startswith("data: "):
                data = line[6:].strip()
                if not data or data == "[DONE]":
                    continue

                try:
                    chunk = json.loads(data)

                    # Stream content in real-time
                    delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        full_content += content
                        log_stream_content(content)

                    # Save usage/timings from ANY chunk that has it (usually the very last one)
                    if "usage" in chunk:
                        final_usage = chunk["usage"]
                    if "timings" in chunk:
                        final_timings = chunk["timings"]

                except json.JSONDecodeError:
                    log_warning(f"Failed to parse JSON: {data}")

        # Always print final newline + summary, even if no usage (still looks clean)
        console.print()  # spacing after streamed text
        log_final_summary(final_usage, final_timings)
        log_info("Stream complete.")

    except requests.exceptions.Timeout:
        log_error("⏰ Timeout: Request timed out after 10 seconds.")
    except requests.exceptions.ConnectionError as e:
        log_error(f"❌ Connection error: {e}")
    except requests.exceptions.RequestException as e:
        log_error(f"❌ Request failed: {e}")
    except Exception as e:
        log_error(f"❌ Unexpected error: {e}")

# Run the script
if __name__ == "__main__":
    main()
