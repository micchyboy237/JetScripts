from ddtrace import config, patch
from jet.logger import logger
import os
import shutil


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
# Datadog Tracing

>[ddtrace](https://github.com/DataDog/dd-trace-py) is a Datadog application performance monitoring (APM) library which provides an integration to monitor your LangChain application.

Key features of the ddtrace integration for LangChain:
- Traces: Capture LangChain requests, parameters, prompt-completions, and help visualize LangChain operations.
- Metrics: Capture LangChain request latency, errors, and token/cost usage (for Ollama LLMs and chat models).
- Logs: Store prompt completion data for each LangChain operation.
- Dashboard: Combine metrics, logs, and trace data into a single plane to monitor LangChain requests.
- Monitors: Provide alerts in response to spikes in LangChain request latency or error rate.

Note: The ddtrace LangChain integration currently provides tracing for LLMs, chat models, Text Embedding Models, Chains, and Vectorstores.

## Installation and Setup

1. Enable APM and StatsD in your Datadog Agent, along with a Datadog API key. For example, in Docker:

docker run -d --cgroupns host \
              --pid host \
              -v /var/run/docker.sock:/var/run/docker.sock:ro \
              -v /proc/:/host/proc/:ro \
              -v /sys/fs/cgroup/:/host/sys/fs/cgroup:ro \
              -e DD_API_KEY=<DATADOG_API_KEY> \
              -p 127.0.0.1:8126:8126/tcp \
              -p 127.0.0.1:8125:8125/udp \
              -e DD_DOGSTATSD_NON_LOCAL_TRAFFIC=true \
              -e DD_APM_ENABLED=true \
              gcr.io/datadoghq/agent:latest

2. Install the Datadog APM Python library.

# pip install ddtrace>=1.17

3. The LangChain integration can be enabled automatically when you prefix your LangChain Python application command with `ddtrace-run`:

DD_SERVICE="my-service" DD_ENV="staging" DD_API_KEY=<DATADOG_API_KEY> ddtrace-run python <your-app>.py

**Note**: If the Agent is using a non-default hostname or port, be sure to also set `DD_AGENT_HOST`, `DD_TRACE_AGENT_PORT`, or `DD_DOGSTATSD_PORT`.

Additionally, the LangChain integration can be enabled programmatically by adding `patch_all()` or `patch(langchain=True)` before the first import of `langchain` in your application.

Note that using `ddtrace-run` or `patch_all()` will also enable the `requests` and `aiohttp` integrations which trace HTTP requests to LLM providers, as well as the `ollama` integration which traces requests to the Ollama library.
"""
logger.info("# Datadog Tracing")



patch(langchain=True)

"""
See the [APM Python library documentation](https://ddtrace.readthedocs.io/en/stable/installation_quickstart.html) for more advanced usage.


## Configuration

See the [APM Python library documentation](https://ddtrace.readthedocs.io/en/stable/integrations.html#langchain) for all the available configuration options.


### Log Prompt & Completion Sampling

To enable log prompt and completion sampling, set the `DD_LANGCHAIN_LOGS_ENABLED=1` environment variable. By default, 10% of traced requests will emit logs containing the prompts and completions.

To adjust the log sample rate, see the [APM library documentation](https://ddtrace.readthedocs.io/en/stable/integrations.html#langchain).

**Note**: Logs submission requires `DD_API_KEY` to be specified when running `ddtrace-run`.


## Troubleshooting

Need help? Create an issue on [ddtrace](https://github.com/DataDog/dd-trace-py) or contact [Datadog support](https://docs.datadoghq.com/help/).
"""
logger.info("## Configuration")

logger.info("\n\n[DONE]", bright=True)