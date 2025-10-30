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
                "content": "Use available tools to calculate arc cosine of 0.5.",
                "role": "user"
            }
        ],
        "model": "qwen3-instruct-2507:4b",
        "stream": True,
        "temperature": 0.0,
        "tools": [
            {
            "type": "function",
            "function": {
                "name": "acos",
                "description": "Return the arc cosine (measured in radians) of x.\n\nThe result is between 0 and pi.",
                "parameters": {
                "properties": {
                    "x": {
                    "type": "number"
                    }
                },
                "required": [
                    "x"
                ],
                "type": "object"
                }
            }
            },
            {
            "type": "function",
            "function": {
                "name": "acosh",
                "description": "Return the inverse hyperbolic cosine of x.",
                "parameters": {
                "properties": {
                    "x": {
                    "type": "number"
                    }
                },
                "required": [
                    "x"
                ],
                "type": "object"
                }
            }
            },
            {
            "type": "function",
            "function": {
                "name": "asin",
                "description": "Return the arc sine (measured in radians) of x.\n\nThe result is between -pi/2 and pi/2.",
                "parameters": {
                "properties": {
                    "x": {
                    "type": "number"
                    }
                },
                "required": [
                    "x"
                ],
                "type": "object"
                }
            }
            },
            {
            "type": "function",
            "function": {
                "name": "asinh",
                "description": "Return the inverse hyperbolic sine of x.",
                "parameters": {
                "properties": {
                    "x": {
                    "type": "number"
                    }
                },
                "required": [
                    "x"
                ],
                "type": "object"
                }
            }
            },
            {
            "type": "function",
            "function": {
                "name": "atan",
                "description": "Return the arc tangent (measured in radians) of x.\n\nThe result is between -pi/2 and pi/2.",
                "parameters": {
                "properties": {
                    "x": {
                    "type": "number"
                    }
                },
                "required": [
                    "x"
                ],
                "type": "object"
                }
            }
            },
            {
            "type": "function",
            "function": {
                "name": "atan2",
                "description": "Return the arc tangent (measured in radians) of y/x.\n\nUnlike atan(y/x), the signs of both x and y are considered.",
                "parameters": {
                "properties": {
                    "y": {
                    "type": "number"
                    },
                    "x": {
                    "type": "number"
                    }
                },
                "required": [
                    "y",
                    "x"
                ],
                "type": "object"
                }
            }
            },
            {
            "type": "function",
            "function": {
                "name": "atanh",
                "description": "Return the inverse hyperbolic tangent of x.",
                "parameters": {
                "properties": {
                    "x": {
                    "type": "number"
                    }
                },
                "required": [
                    "x"
                ],
                "type": "object"
                }
            }
            },
            {
            "type": "function",
            "function": {
                "name": "cbrt",
                "description": "Return the cube root of x.",
                "parameters": {
                "properties": {
                    "x": {
                    "type": "number"
                    }
                },
                "required": [
                    "x"
                ],
                "type": "object"
                }
            }
            },
            {
            "type": "function",
            "function": {
                "name": "ceil",
                "description": "Return the ceiling of x as an Integral.\n\nThis is the smallest integer >= x.",
                "parameters": {
                "properties": {
                    "x": {
                    "type": "number"
                    }
                },
                "required": [
                    "x"
                ],
                "type": "object"
                }
            }
            },
            {
            "type": "function",
            "function": {
                "name": "comb",
                "description": "Number of ways to choose k items from n items without repetition and without order.\n\nEvaluates to n! / (k! * (n - k)!) when k <= n and evaluates\nto zero when k > n.\n\nAlso called the binomial coefficient because it is equivalent\nto the coefficient of k-th term in polynomial expansion of the\nexpression (1 + x)**n.\n\nRaises TypeError if either of the arguments are not integers.\nRaises ValueError if either of the arguments are negative.",
                "parameters": {
                "properties": {
                    "n": {
                    "type": "number"
                    },
                    "k": {
                    "type": "number"
                    }
                },
                "required": [
                    "n",
                    "k"
                ],
                "type": "object"
                }
            }
            },
            {
            "type": "function",
            "function": {
                "name": "copysign",
                "description": "Return a float with the magnitude (absolute value) of x but the sign of y.\n\nOn platforms that support signed zeros, copysign(1.0, -0.0)\nreturns -1.0.\n",
                "parameters": {
                "properties": {
                    "x": {
                    "type": "number"
                    },
                    "y": {
                    "type": "number"
                    }
                },
                "required": [
                    "x",
                    "y"
                ],
                "type": "object"
                }
            }
            },
            {
            "type": "function",
            "function": {
                "name": "cos",
                "description": "Return the cosine of x (measured in radians).",
                "parameters": {
                "properties": {
                    "x": {
                    "type": "number"
                    }
                },
                "required": [
                    "x"
                ],
                "type": "object"
                }
            }
            },
            {
            "type": "function",
            "function": {
                "name": "cosh",
                "description": "Return the hyperbolic cosine of x.",
                "parameters": {
                "properties": {
                    "x": {
                    "type": "number"
                    }
                },
                "required": [
                    "x"
                ],
                "type": "object"
                }
            }
            },
            {
            "type": "function",
            "function": {
                "name": "degrees",
                "description": "Convert angle x from radians to degrees.",
                "parameters": {
                "properties": {
                    "x": {
                    "type": "number"
                    }
                },
                "required": [
                    "x"
                ],
                "type": "object"
                }
            }
            },
            {
            "type": "function",
            "function": {
                "name": "dist",
                "description": "Return the Euclidean distance between two points p and q.\n\nThe points should be specified as sequences (or iterables) of\ncoordinates.  Both inputs must have the same dimension.\n\nRoughly equivalent to:\n    sqrt(sum((px - qx) ** 2.0 for px, qx in zip(p, q)))",
                "parameters": {
                "properties": {
                    "p": {
                    "type": "number"
                    },
                    "q": {
                    "type": "number"
                    }
                },
                "required": [
                    "p",
                    "q"
                ],
                "type": "object"
                }
            }
            },
            {
            "type": "function",
            "function": {
                "name": "erf",
                "description": "Error function at x.",
                "parameters": {
                "properties": {
                    "x": {
                    "type": "number"
                    }
                },
                "required": [
                    "x"
                ],
                "type": "object"
                }
            }
            },
            {
            "type": "function",
            "function": {
                "name": "erfc",
                "description": "Complementary error function at x.",
                "parameters": {
                "properties": {
                    "x": {
                    "type": "number"
                    }
                },
                "required": [
                    "x"
                ],
                "type": "object"
                }
            }
            },
            {
            "type": "function",
            "function": {
                "name": "exp",
                "description": "Return e raised to the power of x.",
                "parameters": {
                "properties": {
                    "x": {
                    "type": "number"
                    }
                },
                "required": [
                    "x"
                ],
                "type": "object"
                }
            }
            },
            {
            "type": "function",
            "function": {
                "name": "exp2",
                "description": "Return 2 raised to the power of x.",
                "parameters": {
                "properties": {
                    "x": {
                    "type": "number"
                    }
                },
                "required": [
                    "x"
                ],
                "type": "object"
                }
            }
            },
            {
            "type": "function",
            "function": {
                "name": "expm1",
                "description": "Return exp(x)-1.\n\nThis function avoids the loss of precision involved in the direct evaluation of exp(x)-1 for small x.",
                "parameters": {
                "properties": {
                    "x": {
                    "type": "number"
                    }
                },
                "required": [
                    "x"
                ],
                "type": "object"
                }
            }
            },
            {
            "type": "function",
            "function": {
                "name": "fabs",
                "description": "Return the absolute value of the float x.",
                "parameters": {
                "properties": {
                    "x": {
                    "type": "number"
                    }
                },
                "required": [
                    "x"
                ],
                "type": "object"
                }
            }
            },
            {
            "type": "function",
            "function": {
                "name": "factorial",
                "description": "Find n!.\n\nRaise a ValueError if x is negative or non-integral.",
                "parameters": {
                "properties": {
                    "n": {
                    "type": "number"
                    }
                },
                "required": [
                    "n"
                ],
                "type": "object"
                }
            }
            },
            {
            "type": "function",
            "function": {
                "name": "floor",
                "description": "Return the floor of x as an Integral.\n\nThis is the largest integer <= x.",
                "parameters": {
                "properties": {
                    "x": {
                    "type": "number"
                    }
                },
                "required": [
                    "x"
                ],
                "type": "object"
                }
            }
            },
            {
            "type": "function",
            "function": {
                "name": "fmod",
                "description": "Return fmod(x, y), according to platform C.\n\nx % y may differ.",
                "parameters": {
                "properties": {
                    "x": {
                    "type": "number"
                    },
                    "y": {
                    "type": "number"
                    }
                },
                "required": [
                    "x",
                    "y"
                ],
                "type": "object"
                }
            }
            },
            {
            "type": "function",
            "function": {
                "name": "frexp",
                "description": "Return the mantissa and exponent of x, as pair (m, e).\n\nm is a float and e is an int, such that x = m * 2.**e.\nIf x is 0, m and e are both 0.  Else 0.5 <= abs(m) < 1.0.",
                "parameters": {
                "properties": {
                    "x": {
                    "type": "number"
                    }
                },
                "required": [
                    "x"
                ],
                "type": "object"
                }
            }
            },
            {
            "type": "function",
            "function": {
                "name": "fsum",
                "description": "Return an accurate floating-point sum of values in the iterable seq.\n\nAssumes IEEE-754 floating-point arithmetic.",
                "parameters": {
                "properties": {
                    "seq": {
                    "type": "number"
                    }
                },
                "required": [
                    "seq"
                ],
                "type": "object"
                }
            }
            },
            {
            "type": "function",
            "function": {
                "name": "gamma",
                "description": "Gamma function at x.",
                "parameters": {
                "properties": {
                    "x": {
                    "type": "number"
                    }
                },
                "required": [
                    "x"
                ],
                "type": "object"
                }
            }
            },
            {
            "type": "function",
            "function": {
                "name": "gcd",
                "description": "Greatest Common Divisor.",
                "parameters": {
                "properties": {
                    "integers": {
                    "type": "number"
                    }
                },
                "required": [
                    "integers"
                ],
                "type": "object"
                }
            }
            },
            {
            "type": "function",
            "function": {
                "name": "isclose",
                "description": "Determine whether two floating-point numbers are close in value.\n\n  rel_tol\n    maximum difference for being considered \"close\", relative to the\n    magnitude of the input values\n  abs_tol\n    maximum difference for being considered \"close\", regardless of the\n    magnitude of the input values\n\nReturn True if a is close in value to b, and False otherwise.\n\nFor the values to be considered close, the difference between them\nmust be smaller than at least one of the tolerances.\n\n-inf, inf and NaN behave similarly to the IEEE 754 Standard.  That\nis, NaN is not close to anything, even itself.  inf and -inf are\nonly close to themselves.",
                "parameters": {
                "properties": {
                    "a": {
                    "type": "number"
                    },
                    "b": {
                    "type": "number"
                    },
                    "rel_tol": {
                    "default": 1e-09,
                    "type": "number"
                    },
                    "abs_tol": {
                    "default": 0.0,
                    "type": "number"
                    }
                },
                "required": [
                    "a",
                    "b"
                ],
                "type": "object"
                }
            }
            },
            {
            "type": "function",
            "function": {
                "name": "isfinite",
                "description": "Return True if x is neither an infinity nor a NaN, and False otherwise.",
                "parameters": {
                "properties": {
                    "x": {
                    "type": "number"
                    }
                },
                "required": [
                    "x"
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