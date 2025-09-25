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
---
id: data-privacy
title: Data Privacy
sidebar_label: Data Privacy
---

<head>
  <link rel="canonical" href="https://deepeval.com/docs/data-privacy" />
</head>

With a mission to ensure consumers are able to be confident in the AI applications they interact with, the team at Confident AI takes data security way more seriously than anyone else.

:::danger
If at any point you think you might have accidentally sent us sensitive data, **please email support@confident-ai.com immediately to request for your data to be deleted.**
:::

## Your Privacy Using DeepEval

By default, `deepeval` uses `Sentry` to track only very basic telemetry data (number of evaluations run and which metric is used). Personally identifiable information is explicitly excluded. We also provide the option of opting out of the telemetry data collection through an environment variable:
"""
logger.info("## Your Privacy Using DeepEval")

export DEEPEVAL_TELEMETRY_OPT_OUT=1

"""
`deepeval` also only tracks errors and exceptions raised within the package **only if you have explicitly opted in**, and **does not collect any user or company data in any way**. To help us catch bugs for future releases, set the `ERROR_REPORTING` environment variable to 1.
"""

export ERROR_REPORTING=1

"""
## Your Privacy Using Confident AI

All data sent to Confident AI is securely stored in databases within our private cloud hosted on AWS (unless your organization is on the VIP plan). **Your organization is the sole entity that can access the data you store.**

We understand that there might still be concerns regarding data security from a compliance point of view. For enhanced security and features, consider upgrading your membership [here.](https://confident-ai.com/pricing)
"""
logger.info("## Your Privacy Using Confident AI")

logger.info("\n\n[DONE]", bright=True)