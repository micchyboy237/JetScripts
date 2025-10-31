#!/usr/bin/env python3
"""
Send a streaming chat completion request to a local server and print each chunk
as it arrives, flushing stdout immediately.
"""

import json
import requests
from typing import Any, Callable
from jet.logger import logger

DEFAULT_PAYLOAD = {
  "messages": [
    {
      "content": "Summarize the full chat history and all tool feedback to\ngive an overview of what the user asked about and what the agent did.",
      "role": "system"
    },
    {
      "content": "Why does RL improve LLM reasoning according to the blogs?",
      "role": "user"
    },
    {
      "content": None,
      "role": "assistant",
      "tool_calls": [
        {
          "type": "function",
          "id": "Hdh7D5f1f1Pv3woJHtZNgWQMTzGAeQaV",
          "function": {
            "name": "retrieve_blog_posts",
            "arguments": "{\"query\": \"how reinforcement learning improves large language model reasoning\"}"
          }
        }
      ]
    },
    {
      "content": "RL improves LLM reasoning by enabling the model to learn optimal policies for generating high-quality reasoning traces (CoTs) through direct reward maximization. Specifically, reinforcement learning (RL) operates within a probabilistic latent variable framework where reasoning steps--formally defined as latent thought variables $z^{(k)}$--serve as intermediate processes that generate correct answers $y$ given questions $x$. The joint distribution is modeled as:\n\n$$\n\\log \\mathcal{L}(\\theta) = \\log \\mathbb{E}_{z^{(k)} \\sim p(z^{(k)} \\mid x)}\\; p(y \\mid z^{(k)}, x)\n$$\n\nThis formulation treats reasoning paths (CoTs) as latent variables, and the goal becomes maximizing the marginal likelihood of correct answers by optimizing both the model parameters $\\theta$ and the distribution over reasoning paths. This is achieved via iterative optimization using the **Expectation-Maximization (EM)** algorithm:\n\n- **E-step (Expectation)**: Sample plausible reasoning traces $z^{(k)}$ given the input $x$, effectively estimating the posterior $p(z^{(k)} \\mid x)$.\n- **M-step (Maximization)**: Update model parameters $\\theta$ to maximize the conditional probability $p(y \\mid z^{(k)}, x)$, i.e., improve the model's ability to produce correct answers given a reasoning path.\n\nThis EM-based training process allows the model to learn to generate reasoning paths that are not only plausible but also directly correlated with correct outputs. The effectiveness of this approach is demonstrated in works such as:\n\n- **Zhou et al. (NeurIPS 2020)**: Introduce explanations as latent variables in natural language understanding, establishing a foundation for reasoning as a latent process.\n- **Phan et al. (NeurIPS 2023)**: Propose training chain-of-thought via latent-variable inference, showing that reasoning traces can be learned and optimized as latent variables.\n- **Ruan et al. (arXiv 2025)**: Introduce \"reasoning to learn from latent thoughts,\" emphasizing that reasoning is a learned latent process that can be refined through structured inference.\n- **Quiet-STaR (Zelikman et al. 2025)**: Implements token-level reasoning by generating rationales after each token, using a **REINFORCE**-based objective to optimize rationale quality. The model alternates between:\n  - **Think**: Predicting next tokens with rationales, using parallel generation and a special attention map to ensure thought tokens only attend to themselves and preceding tokens.\n  - **Talk**: Mixing next-token predictions with and without rationales, using a learned mixing head to dynamically weight the logits.\n  - **Learn**: Training rationale generation via REINFORCE to increase the probability of correct next tokens and reduce incorrect ones, without requiring dataset-specific fine-tuning.\n\nThis framework improves zero-shot performance on CommonsenseQA (36.3% - 47.2%) and GSM8K (5.9% - 10.9%) on Mistral 7B, demonstrating that RL-driven optimization of latent reasoning paths leads to significant gains in reasoning accuracy.\n\nAdditionally, **thinking tokens** and **pause tokens** (e.g., Goyal et al. 2024) provide implicit computational expansion during inference by introducing dummy tokens that increase inference loops without adding new parameters. These tokens act as an implicit form of chain-of-thought reasoning, expanding the model's effective computation. However, their benefits are maximized only when both training and inference involve such tokens, and they do not carry semantic meaning--only computational overhead.\n\nImportantly, the reasoning improvement via RL is not limited to explicit CoT prompting. Instead, it operates at a deeper level by treating reasoning as a latent variable, allowing the model to learn internal reasoning mechanisms through reward-based optimization. This is further supported by **Math-Shepherd (Peiyi Wang et al. 2023)**, which verifies and reinforces LLM reasoning step-by-step without human annotations, using a step-by-step validation mechanism that aligns with RL principles.\n\nIn summary, RL improves LLM reasoning by:\n1. Modeling reasoning paths as latent variables in a probabilistic framework.\n2. Using EM to iteratively refine both reasoning traces and answer generation.\n3. Leveraging REINFORCE-based learning (e.g., in Quiet-STaR) to optimize rationale quality.\n4. Enabling zero-shot performance gains on complex reasoning benchmarks.\n5. Being supported by implicit computational mechanisms (thinking/pause tokens) that expand inference time without parameter growth.\n\nAll technical details--including definitions (e.g., latent variables, CoTs, REINFORCE), mechanisms (E-step/M-step, token-level reasoning), and empirical results (e.g., 36.3% - 47.2% on CommonsenseQA)--are preserved and accurately reflected.",
      "role": "tool",
      "tool_call_id": "Hdh7D5f1f1Pv3woJHtZNgWQMTzGAeQaV"
    },
    {
      "content": "Reinforcement Learning (RL) improves Large Language Model (LLM) reasoning by treating reasoning paths--formally known as Chain-of-Thoughts (CoTs)--as latent variables within a probabilistic framework. Here's a breakdown of how this works, based on the technical insights from the blog posts:\n\n### 1. **Reasoning as a Latent Variable**\nThe core idea is to model reasoning steps ($z^{(k)}$) as intermediate latent variables that generate correct answers ($y$) given a question ($x$). The joint likelihood is expressed as:\n$$\n\\log \\mathcal{L}(\\theta) = \\log \\mathbb{E}_{z^{(k)} \\sim p(z^{(k)} \\mid x)}\\; p(y \\mid z^{(k)}, x)\n$$\nThis formulation treats reasoning traces as internal processes that lead to correct outputs. The goal becomes maximizing the marginal likelihood of correct answers by optimizing both the model parameters ($\\theta$) and the distribution over reasoning paths.\n\n---\n\n### 2. **Expectation-Maximization (EM) Framework**\nRL improves reasoning through an iterative EM process:\n- **E-step (Expectation)**: Sample plausible reasoning traces $z^{(k)}$ given the input $x$, effectively estimating the posterior $p(z^{(k)} \\mid x)$.\n- **M-step (Maximization)**: Update model parameters $\\theta$ to maximize $p(y \\mid z^{(k)}, x)$, i.e., improve the model's ability to produce correct answers given a reasoning path.\n\nThis allows the model to learn reasoning paths that are not only plausible but also directly correlated with correct outputs.\n\n---\n\n### 3. **REINFORCE-Based Optimization (e.g., Quiet-STaR)**\nIn frameworks like **Quiet-STaR**, RL is applied at the token level:\n- **Think Phase**: The model generates next tokens with associated rationales (reasoning traces), using a special attention map so that thought tokens only attend to themselves and earlier tokens.\n- **Talk Phase**: The model mixes predictions with and without rationales, using a learned mixing head to dynamically weight logits.\n- **Learn Phase**: The rationale generation is optimized via **REINFORCE**, a policy gradient method that increases the probability of correct next tokens and reduces incorrect ones. This is done without requiring dataset-specific fine-tuning.\n\nThis leads to significant performance gains:\n- **CommonsenseQA**: 36.3% - 47.2% (on Mistral 7B)\n- **GSM8K**: 5.9% - 10.9%\n\n---\n\n### 4. **Implicit Reasoning via Thinking/Pause Tokens**\nModels like those in **Goyal et al. (2024)** introduce **thinking tokens** and **pause tokens** to implicitly expand inference loops:\n- These tokens act as computational overhead, increasing the number of inference steps without adding new parameters.\n- They enable implicit chain-of-thought reasoning during inference.\n- However, their benefits are maximized only when both training and inference involve such tokens--they do not carry semantic meaning.\n\n---\n\n### 5. **Step-by-Step Validation (e.g., Math-Shepherd)**\nWorks like **Math-Shepherd (Peiyi Wang et al., 2023)** validate reasoning step-by-step without human annotations, aligning with RL principles by reinforcing correct reasoning paths and penalizing incorrect ones.\n\n---\n\n### Summary: Why RL Improves LLM Reasoning\n- **Latent Reasoning Modeling**: Reasoning paths are treated as latent variables, enabling structured learning of internal thought processes.\n- **Iterative Refinement**: EM-based training refines both reasoning traces and answer generation.\n- **Reward-Based Optimization**: REINFORCE and similar methods directly optimize the quality of reasoning traces to improve correctness.\n- **Zero-Shot Performance Gains**: RL-driven reasoning leads to significant improvements on benchmarks like CommonsenseQA and GSM8K.\n- **Implicit Computational Expansion**: Thinking/pause tokens provide a scalable way to simulate reasoning without parameter growth.\n\nIn essence, RL improves LLM reasoning by enabling the model to learn and optimize internal reasoning mechanisms through reward maximization--moving beyond simple prompting to a deeper, structured, and trainable form of reasoning.",
      "role": "assistant"
    }
  ],
  "model": "qwen3-instruct-2507:4b",
  "stream": True,
  "temperature": 0.0,
  "verbosity": "high"
}

def _execute_tool(tool_call: dict[str, Any], available_functions: dict[str, Callable[..., Any]]) -> dict[str, Any]:
    """Run a single tool call and return a tool-response message."""
    func_name = tool_call["function"]["name"]
    args_str = tool_call["function"]["arguments"]
    try:
        args = json.loads(args_str)
    except json.JSONDecodeError:
        logger.red(f"Invalid JSON arguments for {func_name}: {args_str}")
        return {
            "role": "tool",
            "content": json.dumps({"error": "Invalid arguments"}),
            "tool_call_id": tool_call.get("id", ""),
        }

    func = available_functions.get(func_name)
    if not func:
        logger.red(f"Tool {func_name} not implemented")
        return {
            "role": "tool",
            "content": json.dumps({"error": "Tool not found"}),
            "tool_call_id": tool_call.get("id", ""),
        }

    try:
        result = func(**args)
    except Exception as e:
        logger.red(f"Tool {func_name} raised: {e}")
        return {
            "role": "tool",
            "content": json.dumps({"error": str(e)}),
            "tool_call_id": tool_call.get("id", ""),
        }

    logger.green(f"Tool {func_name} → {result}")
    return {
        "role": "tool",
        "content": json.dumps({"result": result}),
        "tool_call_id": tool_call.get("id", ""),
    }

def main() -> None:
    url = "http://shawn-pc.local:8080/v1/chat/completions"

    # ---- local implementations ------------------------------------------------
    import math
    available_functions: dict[str, Callable[..., Any]] = {
        "acos": lambda x: math.acos(float(x)),
        "acosh": lambda x: math.acosh(float(x)),
    }

    payload: dict[str, Any] = DEFAULT_PAYLOAD
    headers = {"Content-Type": "application/json"}

    # ---------- first streaming request (assistant may emit tool_calls) ----------
    with requests.post(url, json=payload, headers=headers, stream=True, timeout=60) as resp:
        resp.raise_for_status()

        tool_calls: list[dict] = []
        current_idx: int | None = None
        accumulated: dict[int, dict[str, str]] = {}

        for line in resp.iter_lines():
            if not line or not line.startswith(b"data: "):
                continue

            data = line[6:]
            if data == b"[DONE]":
                break

            chunk = json.loads(data)
            choices = chunk.get("choices")
            if not choices:
                break

            delta = choices[0].get("delta", {})

            # ---- stream normal content ------------------------------------------------
            if content := delta.get("content"):
                logger.teal(content, flush=True)

            # ---- accumulate tool_call deltas -----------------------------------------
            if tc_deltas := delta.get("tool_calls"):
                for tc in tc_deltas:
                    idx = tc.get("index", 0)
                    if idx not in accumulated:
                        accumulated[idx] = {"id": "", "name": "", "arguments": ""}
                        tool_calls.append({})
                        logger.magenta(f"[Tool {idx}] init", flush=True)

                    if tc_id := tc.get("id"):
                        accumulated[idx]["id"] += tc_id
                        logger.gray(f"[Tool {idx}] id: {tc_id!r}", flush=True)
                    if tc_fn := tc.get("function"):
                        if name := tc_fn.get("name"):
                            accumulated[idx]["name"] += name
                            logger.gray(f"[Tool {idx}] name: {name!r}", flush=True)
                        if args := tc_fn.get("arguments"):
                            accumulated[idx]["arguments"] += args
                            logger.gray(f"[Tool {idx}] args: {args!r}", flush=True)

        # build final tool_call objects
        for idx, acc in accumulated.items():
            tool_calls[idx] = {
                "id": acc["id"],
                "type": "function",
                "function": {"name": acc["name"], "arguments": acc["arguments"]},
            }

    print()  # newline after streaming

    # ---------- if tool calls were emitted, execute them locally -----------------
    if tool_calls:
        messages = payload["messages"][:]
        messages.append({"role": "assistant", "tool_calls": tool_calls})

        for tc in tool_calls:
            messages.append(_execute_tool(tc, available_functions))

        # ---- final non-stream request to get the AI summary --------------------
        final_payload = {
            "model": payload["model"],
            "messages": messages,
            "temperature": payload["temperature"],
        }
        final_resp = requests.post(url, json=final_payload, headers=headers, timeout=60)
        final_resp.raise_for_status()
        final_json = final_resp.json()
        answer = final_json["choices"][0]["message"]["content"]
        logger.cyan("\nFinal AI response:")
        print(answer)
    else:
        logger.yellow("No tool calls emitted.")

if __name__ == "__main__":
    main()