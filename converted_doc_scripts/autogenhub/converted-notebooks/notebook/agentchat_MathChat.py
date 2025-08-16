from autogen.agentchat.contrib.math_user_proxy_agent import MathUserProxyAgent
from jet.logger import CustomLogger
import autogen
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Auto Generated Agent Chat: Using MathChat to Solve Math Problems

AutoGen offers conversable agents powered by LLM, tool or human, which can be used to perform tasks collectively via automated chat. This framework allows tool use and human participation through multi-agent conversation. Please find documentation about this feature [here](https://autogenhub.github.io/autogen/docs/Use-Cases/agent_chat).

MathChat is an experimental conversational framework for math problem solving. In this notebook, we demonstrate how to use MathChat to solve math problems. MathChat uses the `AssistantAgent` and `MathUserProxyAgent`, which is similar to the usage of `AssistantAgent` and `UserProxyAgent` in other notebooks (e.g., [Automated Task Solving with Code Generation, Execution & Debugging](https://github.com/autogenhub/autogen/blob/main/notebook/agentchat_auto_feedback_from_code_execution.ipynb)). Essentially, `MathUserProxyAgent` implements a different auto reply mechanism corresponding to the MathChat prompts. You can find more details in the paper [An Empirical Study on Challenging Math Problem Solving with GPT-4](https://arxiv.org/abs/2306.01337) or the [blogpost](https://autogenhub.github.io/autogen/blog/2023/06/28/MathChat).

````{=mdx}
:::info Requirements
Some extra dependencies are needed for this notebook, which can be installed via pip:

```bash
pip install pyautogen[mathchat]
```

For more information, please refer to the [installation guide](/docs/installation/).
:::
````

## Set your API Endpoint

The [`config_list_from_json`](https://autogenhub.github.io/autogen/docs/reference/oai/openai_utils#config_list_from_json) function loads a list of configurations from an environment variable or a json file.
"""
logger.info("# Auto Generated Agent Chat: Using MathChat to Solve Math Problems")



config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": {
            "gpt-4-1106-preview",
            "gpt-3.5-turbo",
            "gpt-35-turbo",
        }
    },
)

"""
It first looks for environment variable "OAI_CONFIG_LIST" which needs to be a valid json string. If that variable is not found, it then looks for a json file named "OAI_CONFIG_LIST". It filters the configs by models (you can filter by other keys as well).

The config list looks like the following:
```python
config_list = [
    {
        'model': 'gpt-4',
        'api_key': '<your MLX API key here>',
    },
    {
        'model': 'gpt-4',
        'api_key': '<your Azure MLX API key here>',
        'base_url': '<your Azure MLX API base here>',
        'api_type': 'azure',
        'api_version': '2024-02-01',
    },
    {
        'model': 'gpt-3.5-turbo',
        'api_key': '<your Azure MLX API key here>',
        'base_url': '<your Azure MLX API base here>',
        'api_type': 'azure',
        'api_version': '2024-02-01',
    },
]
```

If you open this notebook in colab, you can upload your files by clicking the file icon on the left panel and then choose "upload file" icon.

You can set the value of config_list in other ways you prefer, e.g., loading from a YAML file.

## Construct agents for MathChat

We start by initializing the `AssistantAgent` and `MathUserProxyAgent`. The system message needs to be set to "You are a helpful assistant." for MathChat. The detailed instructions are given in the user message. Later we will use the `MathUserProxyAgent.message_generator` to combine the instructions and a math problem for an initial message to be sent to the LLM assistant.
"""
logger.info("## Construct agents for MathChat")

assistant = autogen.AssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config={
        "timeout": 600,
        "seed": 42,
        "config_list": config_list,
    },
)

mathproxyagent = MathUserProxyAgent(
    name="mathproxyagent",
    human_input_mode="NEVER",
    code_execution_config={"use_docker": False},
)

"""
### Example 1

Problem: Find all $x$ that satisfy the inequality $(2x+10)(x+3)<(3x+9)(x+8)$. Express your answer in interval notation.

Correct Solution: 
We have \begin{align*} (2x+10)(x+3)&<(3x+9)(x+8) \quad \Rightarrow
\\ 2(x+5)(x+3)&<3(x+3)(x+8) \quad \Rightarrow
\\ 2(x+5)(x+3)-3(x+3)(x+8)&<0 \quad \Rightarrow
\\ (2x+10-(3x+24))(x+3)&<0 \quad \Rightarrow
\\ (-x-14)(x+3)&<0 \quad \Rightarrow
\\ (x+14)(x+3)&>0.
\end{align*} This inequality is satisfied if and only if $(x+14)$ and $(x+3)$ are either both positive or both negative.  Both factors are positive for $x>-3$ and both factors are negative for $x<-14$.  When $-14<x<-3$, one factor is positive and the other negative, so their product is negative.   Therefore, the range of $x$ that satisfies the inequality is $ \boxed{(-\infty, -14)\cup(-3,\infty)} $.
"""
logger.info("### Example 1")

math_problem = (
    "Find all $x$ that satisfy the inequality $(2x+10)(x+3)<(3x+9)(x+8)$. Express your answer in interval notation."
)

mathproxyagent.initiate_chat(assistant, message=mathproxyagent.message_generator, problem=math_problem)

"""
### Example 2
Problem: For what negative value of $k$ is there exactly one solution to the system of equations \begin{align*}
y &= 2x^2 + kx + 6 \\
y &= -x + 4?
\end{align*}

Correct Solution: Setting the two expressions for $y$ equal to each other, it follows that $2x^2 + kx + 6 = -x + 4$. Re-arranging, $2x^2 + (k+1)x + 2 = 0$. For there to be exactly one solution for $x$, then the discriminant of the given quadratic must be equal to zero. Thus, $(k+1)^2 - 4 \cdot 2 \cdot 2 = (k+1)^2 - 16 = 0$, so $k+1 = \pm 4$. Taking the negative value, $k = \boxed{-5}$.
"""
logger.info("### Example 2")

math_problem = "For what negative value of $k$ is there exactly one solution to the system of equations \\begin{align*}\ny &= 2x^2 + kx + 6 \\\\\ny &= -x + 4?\n\\end{align*}"
mathproxyagent.initiate_chat(assistant, message=mathproxyagent.message_generator, problem=math_problem)

"""
### Example 3
Problem: Find all positive integer values of $c$ such that the equation $x^2-7x+c=0$ only has roots that are real and rational. Express them in decreasing order, separated by commas.

Correct Solution: For the roots to be real and rational, the discriminant must be a perfect square. Therefore, $(-7)^2-4 \cdot 1 \cdot c = 49-4c$ must be a perfect square. The only positive perfect squares less than 49 are $1$, $4$, $9$, $16$, $25$, and $36$. The perfect squares that give a integer value of $c$ are $1$, $9$, and $25$. Thus, we have the equations $49-4c=1$, $49-4c=9$, and $49-4c=25$. Solving, we get that the positive integer values of c are $\boxed{12, 10, 6}$.
"""
logger.info("### Example 3")

math_problem = "Find all positive integer values of $c$ such that the equation $x^2-7x+c=0$ only has roots that are real and rational. Express them in decreasing order, separated by commas."
mathproxyagent.initiate_chat(assistant, message=mathproxyagent.message_generator, problem=math_problem)

"""
------------------------------------
### Using other prompts

MathChat allows different prompts that instruct the assistant to solve the problem.

Check out `MathUserProxyAgent.message_generator`:
<!-- TODO: Revise this doc accordingly -->
- You may choose from `['default', 'python', 'two_tools']` for parameter `prompt_type`.  We include two more prompts in the paper: 
    1. `'python'` is a simplified prompt from the default prompt that uses Python only. 
    2. `'two_tools'` further allows the selection of Python or Wolfram Alpha based on this simplified `python` prompt. Note that this option requires a Wolfram Alpha API key and put it in `wolfram.txt`.

- You can also input your customized prompt if needed.
Since this mathproxyagent detects '\boxed{}' as termination, you need to have a similar termination sentence in the prompt: "If you get the answer, put the answer in \\boxed{}.". If the customized is provided, the `prompt_type` will be ignored.


### Example 4 (Use the "python" prompt):

Problem: If $725x + 727y = 1500$ and $729x+ 731y = 1508$, what is the value of $x - y$ ?

Correct Solution: Subtracting the two equations gives: 
\begin{align*}
(729x+731y)-(725x+727y) &= 1508-1500\\
\Rightarrow\qquad 4x+4y &= 8\\
\Rightarrow\qquad x+y &= 2.
\end{align*}

Multiplying this equation by 725 and subtracting this equation from $725x+727y=1500$ gives \begin{align*}
(725x+727y) - 725(x+y) &= 1500-725(x+y) \implies \\
2y &= 50.
\end{align*}So we can write $x-y$ as $(x+y) - 2y$, which equals  $2 - 50 = \boxed{-48}$.
"""
logger.info("### Using other prompts")

math_problem = "Problem: If $725x + 727y = 1500$ and $729x+ 731y = 1508$, what is the value of $x - y$ ?"
mathproxyagent.initiate_chat(
    assistant, message=mathproxyagent.message_generator, problem=math_problem, prompt_type="python"
)

"""
## Example 5 (Use the "two_tools" prompt)

Problem: Find all numbers $a$ for which the graph of $y=x^2+a$ and the graph of $y=ax$ intersect. Express your answer in interval notation.


Correct Solution: If these two graphs intersect then the points of intersection occur when  \[x^2+a=ax,\] or  \[x^2-ax+a=0.\] This quadratic has solutions exactly when the discriminant is nonnegative: \[(-a)^2-4\cdot1\cdot a\geq0.\] This simplifies to  \[a(a-4)\geq0.\] This quadratic (in $a$) is nonnegative when $a$ and $a-4$ are either both $\ge 0$ or both $\le 0$. This is true for $a$ in $$(-\infty,0]\cup[4,\infty).$$ Therefore the line and quadratic intersect exactly when $a$ is in $\boxed{(-\infty,0]\cup[4,\infty)}$.
"""
logger.info("## Example 5 (Use the "two_tools" prompt)")

if "WOLFRAM_ALPHA_APPID" not in os.environ:
    os.environ["WOLFRAM_ALPHA_APPID"] = open("wolfram.txt").read().strip()

math_problem = "Find all numbers $a$ for which the graph of $y=x^2+a$ and the graph of $y=ax$ intersect. Express your answer in interval notation."
mathproxyagent.initiate_chat(
    assistant, message=mathproxyagent.message_generator, problem=math_problem, prompt_type="two_tools"
)

logger.info("\n\n[DONE]", bright=True)