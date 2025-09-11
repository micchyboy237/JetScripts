from jet.logger import logger
import graphsignal
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
# Graphsignal

This page covers how to use [Graphsignal](https://app.graphsignal.com) to trace and monitor LangChain. Graphsignal enables full visibility into your application. It provides latency breakdowns by chains and tools, exceptions with full context, data monitoring, compute/GPU utilization, Ollama cost analytics, and more.

## Installation and Setup

- Install the Python library with `pip install graphsignal`
- Create free Graphsignal account [here](https://graphsignal.com)
- Get an API key and set it as an environment variable (`GRAPHSIGNAL_API_KEY`)

## Tracing and Monitoring

Graphsignal automatically instruments and starts tracing and monitoring chains. Traces and metrics are then available in your [Graphsignal dashboards](https://app.graphsignal.com).

Initialize the tracer by providing a deployment name:
"""
logger.info("# Graphsignal")


graphsignal.configure(deployment='my-langchain-app-prod')

"""
To additionally trace any function or code, you can use a decorator or a context manager:
"""
logger.info("To additionally trace any function or code, you can use a decorator or a context manager:")

@graphsignal.trace_function
def handle_request():
    chain.run("some initial text")

"""

"""

with graphsignal.start_trace('my-chain'):
    chain.run("some initial text")

"""
Optionally, enable profiling to record function-level statistics for each trace.
"""
logger.info("Optionally, enable profiling to record function-level statistics for each trace.")

with graphsignal.start_trace(
        'my-chain', options=graphsignal.TraceOptions(enable_profiling=True)):
    chain.run("some initial text")

"""
See the [Quick Start](https://graphsignal.com/docs/guides/quick-start/) guide for complete setup instructions.
"""
logger.info("See the [Quick Start](https://graphsignal.com/docs/guides/quick-start/) guide for complete setup instructions.")

logger.info("\n\n[DONE]", bright=True)