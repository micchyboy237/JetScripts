import argparse
import os

from jet.libs.llama_cpp.utils.performance_tracker import PerformanceTracker, log_metrics
from jet.logger import logger
from openai import OpenAI, Stream
from openai.types.chat import ChatCompletionChunk

client = OpenAI(
    base_url=os.getenv("LLAMA_CPP_LLM_URL", "http://localhost:1234/v1"),
    api_key="sk-1234",
)


def run_chat_stream(
    user_prompt: str, system_prompt: str | None = None, verbose: bool = False
):
    messages = []
    if system_prompt:
        messages.append(
            {
                "role": "system",
                "content": system_prompt,
            }
        )
    messages.append(
        {
            "role": "user",
            "content": user_prompt,
        }
    )

    if system_prompt:
        if verbose:
            logger.log("System prompt: ", system_prompt, colors=["PURPLE", "DEBUG"])
    if verbose:
        logger.log("User prompt: ", user_prompt, colors=["GRAY", "DEBUG"])

    tracker = PerformanceTracker()

    stream: Stream[ChatCompletionChunk] = client.chat.completions.create(
        model="Qwen/Qwen3.5-2B",
        messages=messages,
        max_tokens=1024,
        temperature=1.0,
        top_p=1.0,
        presence_penalty=2.0,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {
                "enable_thinking": False,
            },
        },
        stream=True,
    )

    content = ""
    for part in stream:
        if part.choices and part.choices[0].delta:
            delta = part.choices[0].delta

            # Check for reasoning_content first
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                content += delta.reasoning_content
                tracker.mark_token()
                if verbose:
                    logger.orange(delta.reasoning_content, flush=True, end="")
            # Then check for regular content
            elif hasattr(delta, "content") and delta.content:
                content += delta.content
                tracker.mark_token()
                if verbose:
                    logger.teal(delta.content, flush=True, end="")

        usage = getattr(part, "usage", None)
        if usage is not None:
            metrics = tracker.finalize(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
            )

            if verbose:
                log_metrics(metrics)

    return content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stream chat completion from llama.cpp OpenAI API-compatible endpoint"
    )
    parser.add_argument(
        "prompt",
        type=str,
        nargs="?",
        default="Write a 2 sentence short story about a curious robot.",
        help="User input prompt for the chat model (default: %(default)s)",
    )
    parser.add_argument(
        "-s",
        "--system",
        type=str,
        default=None,
        help="Optional system prompt for the chat model",
    )
    args = parser.parse_args()

    user_prompt = args.prompt
    system_prompt = args.system

    run_chat_stream(user_prompt, system_prompt, verbose=True)
