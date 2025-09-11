from jet.logger import logger
from langchain_community.utilities import SearxSearchWrapper
import os
import pprint
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
# SearxNG Search

This notebook goes over how to use a self hosted `SearxNG` search API to search the web.

You can [check this link](https://docs.searxng.org/dev/search_api.html) for more informations about `Searx API` parameters.
"""
logger.info("# SearxNG Search")



search = SearxSearchWrapper(searx_host="http://127.0.0.1:8888")

"""
For some engines, if a direct `answer` is available the warpper will print the answer instead of the full list of search results. You can use the `results` method of the wrapper if you want to obtain all the results.
"""
logger.info("For some engines, if a direct `answer` is available the warpper will print the answer instead of the full list of search results. You can use the `results` method of the wrapper if you want to obtain all the results.")

search.run("What is the capital of France")

"""
## Custom Parameters

SearxNG supports [135 search engines](https://docs.searxng.org/user/configured_engines.html). You can also customize the Searx wrapper with arbitrary named parameters that will be passed to the Searx search API . In the below example we will making a more interesting use of custom search parameters from searx search api.

In this example we will be using the `engines` parameters to query wikipedia
"""
logger.info("## Custom Parameters")

search = SearxSearchWrapper(
    searx_host="http://127.0.0.1:8888", k=5
)  # k is for max number of items

search.run("large language model ", engines=["wiki"])

"""
Passing other Searx parameters for searx like `language`
"""
logger.info("Passing other Searx parameters for searx like `language`")

search = SearxSearchWrapper(searx_host="http://127.0.0.1:8888", k=1)
search.run("deep learning", language="es", engines=["wiki"])

"""
## Obtaining results with metadata

In this example we will be looking for scientific paper using the `categories` parameter and limiting the results to a `time_range` (not all engines support the time range option).

We also would like to obtain the results in a structured way including metadata. For this we will be using the `results` method of the wrapper.
"""
logger.info("## Obtaining results with metadata")

search = SearxSearchWrapper(searx_host="http://127.0.0.1:8888")

results = search.results(
    "Large Language Model prompt",
    num_results=5,
    categories="science",
    time_range="year",
)
pprint.pp(results)

"""
Get papers from arxiv
"""
logger.info("Get papers from arxiv")

results = search.results(
    "Large Language Model prompt", num_results=5, engines=["arxiv"]
)
pprint.pp(results)

"""
In this example we query for `large language models` under the `it` category. We then filter the results that come from github.
"""
logger.info("In this example we query for `large language models` under the `it` category. We then filter the results that come from github.")

results = search.results("large language model", num_results=20, categories="it")
pprint.pp(list(filter(lambda r: r["engines"][0] == "github", results)))

"""
We could also directly query for results from `github` and other source forges.
"""
logger.info("We could also directly query for results from `github` and other source forges.")

results = search.results(
    "large language model", num_results=20, engines=["github", "gitlab"]
)
pprint.pp(results)

logger.info("\n\n[DONE]", bright=True)