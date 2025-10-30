#!/usr/bin/env python3
"""
Send a streaming chat completion request to a local server and print each chunk
as it arrives, flushing stdout immediately.
"""

import json
import requests
from typing import Any, Callable
from jet.logger import logger

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

    payload: dict[str, Any] = {
        "messages": [
            {
                "content": "You are a helpful assistant tasked with retrieving information from a series of technical blog posts by Lilian Weng. \nClarify the scope of research with the user before using your retrieval tool to gather context. Reflect on any context you fetch, and\nproceed until you have sufficient context to answer the user's research request.\n\nContext:\nDocuments:\nLanguage-Model\nReinforcement-Learning\nReasoning\nLong-Read\n\n\n\n >>\n\nReward Hacking in Reinforcement Learning\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n(c) 2025 Lil'Log\n\n        Powered by\n        Hugo &\n        PaperMod\n\nExperiments in two RL environments, CoinRun and Maze, demonstrated the importance of randomization during training. If during training, the coin or the cheese is placed at a fixed position (i.e. right end of the level or upper right corner of the maze) but testing in the env where the coin or cheese is placed at random, the agent would just run to the fixed position without obtaining the coin or cheese at test time. A conflict arises when a visual feature (e.g., cheese or coin) and a positional feature (e.g., upper-right or right end) are inconsistent during test time, leading the trained model to prefer the positional feature. I would like to point out that, in these two examples, the reward-result gaps are clear but such type of biases are unlikely to be so obvious in most real-world cases.\n\n\nThe impact of randomizing the position of the coin during training. When the coin is placed at random for {0, 2, 3, 6, 11}% of the time during training (x-axis), the frequency of the agent navigating to the end of the level without obtaining the coin decreases with the increase of the randomization (\"y-axis\"). (Image source: Koch et al. 2021)\n\nReward Tampering (Everitt et al. 2019) is a form of reward hacking behavior where the agent interferes with the reward function itself, causing the observed reward to no longer accurately represent the intended goal. In reward tampering, the model modifies its reward mechanism either by directly manipulating the implementation of the reward function or by indirectly altering the environmental information used as input for the reward function.\n(Note: Some work defines reward tampering as a distinct category of misalignment behavior from reward hacking. But I consider reward hacking as a broader concept here.)\nAt a high level, reward hacking can be categorized into two types: environment or goal misspecification, and reward tampering.\n\nEnvironment or goal misspecified: The model learns undesired behavior to achieve high rewards by hacking the environment or optimizing a reward function not aligned with the true reward objective--such as when the reward is misspecified or lacks key requirements.\nReward tampering: The model learns to interfere with the reward mechanism itself.\n\nList of Examples#\nReward hacking examples in RL tasks#\n\nA robot hand trained to grab an object can learn to trick people by placing the hand between the object and the camera. (Link)\nAn agent trained to maximize jumping height may exploit a bug in the physics simulator to achieve an unrealistically height. (Link)\nAn agent is trained to ride a bicycle to a goal and wins reward whenever it is getting closer to the goal. Then the agent may learn to ride in tiny circles around the goal because there is no penalty when the agent gets away from the goal. (Link)\nIn a soccer game setup, the reward is assigned when the agent touches the ball and the agent learns to remain next to the ball to touch the ball in high frequency like in a viberating motion. (Link)\nIn the Coast Runners game, an agent controls a boat with the goal to finish the boat race as quickly as possible. When it is given a shaping reward for hitting green blocks along the race track, it changes the optimal policy to going in circles and hitting the same green blocks over and over again. (Link)\n\"The Surprising Creativity of Digital Evolution\"  (Lehman et al. 2019) - This paper has many examples about how optimizing a misspecified fitness function can lead to surprising \"hacking\" or unintended evolutionary or learning results.\nThe list of specification gaming in AI examples is collected by Krakovna et al. 2020.\n\nReward hacking examples in LLM tasks#\n\nA language model for generating summarization is able to explore flaws in the ROUGE metric such that it obtains high score but the generated summaries are barely readable. (Link)\nA coding model learns to change unit test in order to pass coding questions. (Link)\nA coding model may learn to directly modify the code used for calculating the reward. (Link)\n\nReward hacking examples in real life#\n\nThe recommendation algorithm for social media is intended to provide useful information. However, usefulness is often measured by proxy metrics, such as the number of likes or comments, or the time or frequency of engagement on the platform. The algorithm ends up recommending content that can affect users' emotion states such as outrageous and extreme content in order to trigger more engagement. (Harari, 2024)\nOptimizing for misspecified proxy metrics for a video sharing site may aggressively increase the watch time of users while the true goal is to optimize users' subjective well-being. (Link)\n\"The Big Short\" - 2008 financial crisis caused by the housing bubble. Reward hacking of our society happened as people tried to game the financial system.\n\nWhy does Reward Hacking Exist?#\nGoodhart's Law states that \"When a measure becomes a target, it ceases to be a good measure\". The intuition is that a good metric can become corrupted once significant pressure is applied to optimize it. It is challenging to specify a 100% accurate reward objective and any proxy suffers the risk of being hacked, as RL algorithm exploits any small imperfection in the reward function definition. Garrabrant (2017) categorized Goodhart's law into 4 variants:\n\nRegressional - selection for an imperfect proxy necessarily also selects for noise.\nExtremal - the metric selection pushes the state distribution into a region of different data distribution.\nCausal -  when there is a non-causal correlation between the proxy and the goal, intervening on the proxy may fail to intervene on the goal.\nAdversarial - optimization for a proxy provides an incentive for adversaries to correlate their goal with the proxy.\n\nAmodei et al. (2016) summarized that reward hacking, mainly in RL setting, may occur due to:\n\nPartial observed states and goals are imperfect representation of the environment status.\nThe system itself is complex and susceptible to hacking; e.g., if the agent is allowed to execute code that changes part of the environment, it becomes much easier to exploit the environment's mechanisms.\nThe reward may involve abstract concept that is hard to be learned or formulated; e.g., a reward function with high-dimensional inputs may disproportionately rely on a few dimensions.\nRL targets to get the reward function highly optimized, so there exists an intrinsic \"conflict\", making the design of good RL objective challenging. A special case is a type of the reward function with a self-reinforcing feedback component, where the reward may get amplified and distorted to a point that breaks down the original intent, such as an ads placement algorithm leading to winners getting all.\n\nBesides, identifying the exact reward function for which an optimal agent optimizes its behavior is in general impossible since there could be an infinite number of reward functions consistent with any observed policy in an fixed environment (Ng & Russell, 2000). Amin and Singh (2016) separated the causes of this unidentifiability into two classes:\n\nRepresentational - a set of reward functions is behaviorally invariant under certain arithmetic operations (e.g., re-scaling)\nExperimental - $\\pi$'s observed behavior is insufficient to distinguish between two or more reward functions which both rationalize the behavior of the agent (the behavior is optimal under both)\n\nHistory:\n\nQuestion: What are the types of reward hacking discussed in the blogs?",
                "role": "system"
            }
        ],
        "model": "qwen3-instruct-2507:4b",
        "stream": True,
        "temperature": 0.0,
        "tools": [
            {
            "type": "function",
            "function": {
                "name": "retrieve_blog_posts",
                "description": "Search and return information about Lilian Weng blog posts.",
                "parameters": {
                "properties": {
                    "query": {
                    "description": "query to look up in retriever",
                    "type": "string"
                    }
                },
                "required": [
                    "query"
                ],
                "type": "object"
                }
            }
            }
        ],
    }
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