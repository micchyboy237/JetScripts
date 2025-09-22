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
# Workflows

A [`Workflow`](/python/workflows/) in LlamaIndex is an event-driven abstraction used to chain together several events. Workflows are made up
of `steps`, with each step responsible for handling certain event types and emitting new events.

You can create a `Workflow` to do anything! Build an agent, a RAG flow, an extraction flow, or anything else you want.

Workflows are also automatically instrumented, so you get observability into each step using tools like [Arize Pheonix](/python/framework/module_guides/observability#arize-phoenix-local). (**NOTE:** Observability works for integrations that take advantage of the newer instrumentation system. Usage may vary.)


<Aside>
The Workflows library can be installed standalone, via `pip install llama-index-workflows`. However,
`llama-index-core` comes with a stable version of Workflows included.
</Aside>

    When installing `llama-index-core` or the `llama-index` umbrella package, Workflows can be accessed with the import
    path `llama_index.core.workflow`. In order to maintain the `llama_index` API stable and avoid breaking changes,
    the Workflows library version included is usually older than the latest version of `llama-index-workflows`.

    At the moment, the latest version of `llama-index-workflows` is 2.0 while the one shipped with `llama-index` is
    1.3

- [v1.x Documentation](/python/workflows/v1/)
- [v2.x Documentation](/python/workflows/v2/)
"""
logger.info("# Workflows")

logger.info("\n\n[DONE]", bright=True)