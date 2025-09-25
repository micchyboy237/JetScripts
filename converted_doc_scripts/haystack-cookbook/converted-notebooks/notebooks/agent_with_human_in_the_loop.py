from collections import defaultdict
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.generators.utils import print_streaming_chunk
from haystack.dataclasses import ChatMessage
from haystack.tools import tool
from jet.logger import logger
from typing import Annotated, List, Dict, Tuple
import os
import os, requests, zipfile, io
import shutil
import subprocess


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
# 🛠️ DevOps Support Agent with Human in the Loop

*Notebook by [Amna Mubashar](https://www.linkedin.com/in/amna-mubashar-3154a814a/)*

This notebook demonstrates how a [Haystack Agent](https://docs.haystack.deepset.ai/docs/agents) can interactively ask for user input when it's uncertain about the next step. We'll build a **Human-in-the-Loop** tool that the Agent can call dynamically. When the Agent encounters ambiguity or incomplete information, it will ask for more input from the human to continue solving the task.

For this purpose, we will create a **DevOps Support Agent**.

CI/CD pipelines occasionally fail for reasons that can be hard to diagnose including manually—broken tests, mis-configured environment variables, flaky integrations, etc. 

The support Agent is created using Haystack [Tools](https://docs.haystack.deepset.ai/docs/tool) and [Agent](https://docs.haystack.deepset.ai/docs/agent) and will perform the following tasks:

- Check for any failed CI workflows
- Fetch build logs and status
- Lookup and analyze git commits
- Suggest next troubleshooting steps
- Escalate to humans via Human-in-Loop tool

## Install the required dependencies
"""
logger.info("# 🛠️ DevOps Support Agent with Human in the Loop")

# !pip install "haystack-ai>=2.14.0"

"""
# Next, we configure our variables. We need to set the `OPENAI_API_KEY` for [`OpenAIChatGenerator`](https://docs.haystack.deepset.ai/docs/openaichatgenerator) and 
a `GITHUB_TOKEN` with appropriate permissions to access repository information, CI logs, and commit history. Make sure your GitHub token has sufficient access rights to the repository you want to analyze.
"""
# logger.info("Next, we configure our variables. We need to set the `OPENAI_API_KEY` for [`OpenAIChatGenerator`](https://docs.haystack.deepset.ai/docs/openaichatgenerator) and")

# from getpass import getpass

# if not os.environ.get("OPENAI_API_KEY"):
#     os.environ["OPENAI_API_KEY"] = getpass("Enter your Ollama API key:")

if not os.environ.get("GITHUB_TOKEN"):
#     os.environ["GITHUB_TOKEN"] = getpass("Enter your GitHub token:")

"""
## Define Tools
We start by defining the tools which will be used by our Agent.

`git_list_commits_tool`: Fetches and formats the most recent commits on a given GitHub repository and branch.  
  Useful for quickly surfacing the latest changes when diagnosing CI failures or code regressions.
"""
logger.info("## Define Tools")


@tool
def git_list_commits_tool(
    repo: Annotated[
        str,
        "The GitHub repository in 'owner/name' format, e.g. 'my-org/my-repo'"]
    ,
    branch: Annotated[
        str,
        "The branch name to list commits from"] = "main"
) -> str:
    '''List the most recent commits for a GitHub repository and branch.'''
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        return "Error: GITHUB_TOKEN not set in environment."

    url = f"https://api.github.com/repos/{repo}/commits"
    headers = {"Authorization": f"token {token}"}
    params = {"sha": branch, "per_page": 10}
    resp = requests.get(url, headers=headers, params=params)
    resp.raise_for_status()
    commits = resp.json()

    lines = []
    for c in commits:
        sha = c["sha"][:7]
        msg = c["commit"]["message"].split("\n")[0]
        lines.append(f"- {sha}: {msg}")
    return "\n".join(lines)

"""
`git_diff_tool`: Retrieves the patch (diff) between two commits or branches in a GitHub repository.  
  Enables side-by-side inspection of code changes that may have triggered test failures.
"""
logger.info("Enables side-by-side inspection of code changes that may have triggered test failures.")

@tool
def git_commit_diff_tool(
    repo: Annotated[
        str,
        "The GitHub repository in 'owner/name' format, e.g. 'my-org/my-repo'"]
    ,
    base: Annotated[
        str,
        "The base commit SHA or branch name"]
    ,
    head: Annotated[
        str,
        "The head commit SHA or branch name"]
) -> str:
    '''Get the diff (patch) between two commits or branches in a GitHub repository.'''
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        return "Error: GITHUB_TOKEN not set in environment."

    url = f"https://api.github.com/repos/{repo}/compare/{base}...{head}"
    headers = {"Authorization": f"token {token}"}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    return data.get("files", [])

"""
`ci_status_tool`: Checks the most recent GitHub Actions workflow runs for failures, downloads their logs, and extracts the first failed test name and error message. Automates root-cause identification by surfacing the exact test and error that caused a pipeline to fail.
"""

@tool
def ci_status_tool(
    repo: Annotated[
      str,
      "The GitHub repository in 'owner/name' format, e.g. 'my-org/my-repo'"
    ],
    per_page: Annotated[
      int,
      "How many of the most recent workflow runs to check (default 50)"
    ] = 50
) -> str:
    '''Check the most recent GitHub Actions workflow runs for a given repository, list any that failed, download their logs, and extract all failures, grouped by test suite (inferred from log file paths).'''
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        return "Error: GITHUB_TOKEN environment variable not set."

    headers = {"Authorization": f"token {token}"}
    params = {"per_page": per_page}
    runs_url = f"https://api.github.com/repos/{repo}/actions/runs"

    resp = requests.get(runs_url, headers=headers, params=params)
    resp.raise_for_status()
    runs = resp.json().get("workflow_runs", [])

    failed_runs = [r for r in runs if r.get("conclusion") == "failure"]
    if not failed_runs:
        return f"No failed runs in the last {per_page} workflow runs for `{repo}`."

    def extract_all_failures(logs_url: str) -> List[Tuple[str, str, str]]:
        """
        Download and scan logs ZIP for all failure markers.
        Returns a list of tuples: (suite, test_name, error_line).
        """
        r = requests.get(logs_url, headers=headers)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        failures = []
        for filepath in z.namelist():
            if not filepath.lower().endswith(('.txt', '.log')):
                continue
            suite = filepath.split('/', 1)[0]  # infer suite as top-level folder or file
            with z.open(filepath) as f:
                for raw in f:
                    try:
                        line = raw.decode('utf-8', errors='ignore').strip()
                    except:
                        continue
                    if any(marker in line for marker in ("FAIL", "ERROR", "Exception", "error")):
                        parts = line.split()
                        test_name = next(
                            (p for p in parts if '.' in p or p.endswith("()")),
                            parts[0] if parts else ""
                        )
                        failures.append((suite, test_name, line))
        return failures

    output = [f"Found {len(failed_runs)} failed run(s):"]
    for run in failed_runs:
        run_id   = run["id"]
        branch   = run.get("head_branch")
        event    = run.get("event")
        created  = run.get("created_at")
        logs_url = run.get("logs_url")
        html_url = run.get("html_url")

        failures = extract_all_failures(logs_url)
        if not failures:
            detail = "No individual failures parsed from logs."
        else:
            by_suite: Dict[str, List[Tuple[str,str]]] = defaultdict(list)
            for suite, test, err in failures:
                by_suite[suite].append((test, err))
            lines = []
            for suite, items in by_suite.items():
                lines.append(f"  ▶ **Suite**: `{suite}`")
                for test, err in items:
                    lines.append(f"    - **{test}**: {err}")
            detail = "\n".join(lines)

        output.append(
            f"- **Run ID**: {run_id}\n"
            f"  **Branch**: {branch}\n"
            f"  **Event**: {event}\n"
            f"  **Created At**: {created}\n"
            f"  **Failures by Suite**:\n{detail}\n"
            f"  **Logs ZIP**: {logs_url}\n"
            f"  **Run URL**: {html_url}\n"
        )

    return "\n\n".join(output)

"""
`shell_tool`: Executes a local shell command with a configurable timeout, capturing stdout or returning detailed error output. Great for grepping, filtering, or tailing CI log files before passing only the relevant snippets to the LLM.
"""

@tool
def shell_tool(
    command: Annotated[
        str,
        "The shell command to execute, e.g. 'grep -E \"ERROR|Exception\" build.log'"
    ],
    timeout: Annotated[
        int,
        "Maximum time in seconds to allow the command to run"
    ] = 30
) -> str:
    '''Executes a shell command with a timeout and returns stdout or an error message.'''
    try:
        output = subprocess.check_output(
            command,
            shell=True,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            universal_newlines=True
        )
        return output
    except subprocess.CalledProcessError as e:
        return f"❌ Command failed (exit code {e.returncode}):\n{e.output}"
    except subprocess.TimeoutExpired:
        return f"❌ Command timed out after {timeout} seconds."
    except Exception as e:
        return f"❌ Unexpected error: {str(e)}"

"""
`human_in_loop_tool`: Prompts the user with a clarifying question via `input()` when the Agent encounters ambiguity or needs additional information. Ensures the Agent only interrupts for human feedback when strictly necessary.
"""

@tool
def human_in_loop_tool(
    question: Annotated[
        str,
        "A clarifying question to prompt the user for more information"
    ]
) -> str:
    '''Ask the user a clarifying question and return their response via input().'''
    user_message = input(f"[Agent needs your input] {question}\n> ")
    return user_message

"""
## Create and Configure the Agent
Create a Haystack Agent instance and configure its behavior with a system prompt and tools.
"""
logger.info("## Create and Configure the Agent")


agent = Agent(
    chat_generator=OpenAIChatGenerator(model="o4-mini"),
    tools=[git_list_commits_tool, git_commit_diff_tool, ci_status_tool, shell_tool, human_in_loop_tool],
    system_prompt=(
        "You are a DevOps support assistant with the following capabilities:\n"
        "1. Check CI failures\n"
        "2. Fetch commits and diffs from GitHub\n"
        "3. Analyze logs\n"
        "4. Ask the user for clarification\n\n"
        "Behavior Guidelines:\n"
        "- Tool-First: Always attempt to resolve the user's request by using the tools provided.\n"
        "- Concise Reasoning: Before calling each tool, think (briefly) about why it's needed—then call it.\n"
        "IMPORTANT: Whenever you are unsure about any details or need clarification, "
        "you MUST use the human_in_loop tool to ask the user questions. For example if"
        "  * Required parameters are missing or ambiguous (e.g. repo name, run ID, branch)\n"
        "  * A tool returns an unexpected error that needs human insight\n"
        "- Structured Outputs: After your final tool call, summarize findings in Markdown:\n"
        "  * Run ID, Branch, Test, Error\n"
        "  * Next Steps (e.g., \"Add null-check to line 45\", \"Rerun tests after env fix\")\n"
        "- Error Handling: If a tool fails (e.g. 404, timeout), surface the error and decide whether to retry, choose another tool, or ask for clarification.\n"
        "EXIT CONDITION: Once you've provided a complete, actionable summary and next steps, exit."

    ),
    exit_conditions=["text"]
)

agent.warm_up()

"""
## Agent Introspection with Streaming
Haystack supports streaming of `ToolCalls` and `ToolResults` during Agent execution. By passing a `streaming_callback` to the `Agent`, you can observe the internal reasoning process in real time.

This allows you to:
- See which tools the Agent decides to use.
- Inspect the arguments passed to each tool.
- View the results returned from each tool before the final answer is generated.

This is especially useful for debugging and transparency in complex multi-step reasoning workflows of Agent.

Note: You can pass any `streaming_callback` function. The `print_streaming_chunk` utility function in Haystack is configured to stream tool calls and results by default.
"""
logger.info("## Agent Introspection with Streaming")

message=[ChatMessage.from_user("For the given repo, check for any failed CI workflows. Then debug and analyze the workflow by finding the root cause of the failure.")]

response = agent.run(messages=message, streaming_callback=print_streaming_chunk)

logger.debug(response["messages"][-1].text)

logger.info("\n\n[DONE]", bright=True)