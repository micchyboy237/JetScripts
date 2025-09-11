from jet.logger import logger
import People from "@theme/People";
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
---
hide_table_of_contents: true
---


# People

There are some incredible humans from all over the world who have been instrumental in helping the LangChain community flourish 🌐!

This page highlights a few of those folks who have dedicated their time to the open-source repo in the form of direct contributions and reviews.

## Top reviewers

As LangChain has grown, the amount of surface area that maintainers cover has grown as well.

Thank you to the following folks who have gone above and beyond in reviewing incoming PRs 🙏!

<People type="top_reviewers"></People>

## Top recent contributors

The list below contains contributors who have had the most PRs merged in the last three months, weighted (imperfectly) by impact.

Thank you all so much for your time and efforts in making LangChain better ❤️!

<People type="top_recent_contributors" count="20"></People>

## Core maintainers

Hello there 👋!

We're LangChain's core maintainers. If you've spent time in the community, you've probably crossed paths
with at least one of us already. 

<People type="maintainers"></People>

## Top all-time contributors

And finally, this is an all-time list of all-stars who have made significant contributions to the framework 🌟:

<People type="top_contributors"></People>

We're so thankful for your support!

And one more thank you to [@tiangolo](https://github.com/tiangolo) for inspiration via FastAPI's [excellent people page](https://fastapi.tiangolo.com/fastapi-people).
"""
logger.info("# People")

logger.info("\n\n[DONE]", bright=True)