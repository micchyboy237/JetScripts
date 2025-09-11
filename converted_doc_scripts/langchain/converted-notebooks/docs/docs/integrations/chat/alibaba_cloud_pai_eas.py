from jet.logger import logger
from langchain_community.chat_models import PaiEasChatEndpoint
from langchain_core.language_models.chat_models import HumanMessage
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
sidebar_label: Alibaba Cloud PAI EAS
---

# Alibaba Cloud PAI EAS

>[Alibaba Cloud PAI (Platform for AI)](https://www.alibabacloud.com/help/en/pai/?spm=a2c63.p38356.0.0.c26a426ckrxUwZ) is a lightweight and cost-efficient machine learning platform that uses cloud-native technologies. It provides you with an end-to-end modelling service. It accelerates model training based on tens of billions of features and hundreds of billions of samples in more than 100 scenarios.

>[Machine Learning Platform for AI of Alibaba Cloud](https://www.alibabacloud.com/help/en/machine-learning-platform-for-ai/latest/what-is-machine-learning-pai) is a machine learning or deep learning engineering platform intended for enterprises and developers. It provides easy-to-use, cost-effective, high-performance, and easy-to-scale plug-ins that can be applied to various industry scenarios. With over 140 built-in optimization algorithms, `Machine Learning Platform for AI` provides whole-process AI engineering capabilities including data labelling (`PAI-iTAG`), model building (`PAI-Designer` and `PAI-DSW`), model training (`PAI-DLC`), compilation optimization, and inference deployment (`PAI-EAS`).
>
>`PAI-EAS` supports different types of hardware resources, including CPUs and GPUs, and features high throughput and low latency. It allows you to deploy large-scale complex models with a few clicks and perform elastic scale-ins and scale-outs in real-time. It also provides a comprehensive O&M and monitoring system.

## Setup EAS Service

Set up environment variables to init EAS service URL and token.
Use [this document](https://www.alibabacloud.com/help/en/pai/user-guide/service-deployment/) for more information.

```bash
export EAS_SERVICE_URL=XXX
export EAS_SERVICE_TOKEN=XXX
```
Another option is to use this code:
"""
logger.info("# Alibaba Cloud PAI EAS")



os.environ["EAS_SERVICE_URL"] = "Your_EAS_Service_URL"
os.environ["EAS_SERVICE_TOKEN"] = "Your_EAS_Service_Token"
chat = PaiEasChatEndpoint(
    eas_service_url=os.environ["EAS_SERVICE_URL"],
    eas_service_token=os.environ["EAS_SERVICE_TOKEN"],
)

"""
## Run Chat Model

You can use the default settings to call EAS service as follows:
"""
logger.info("## Run Chat Model")

output = chat.invoke([HumanMessage(content="write a funny joke")])
logger.debug("output:", output)

"""
Or, call EAS service with new inference params:
"""
logger.info("Or, call EAS service with new inference params:")

kwargs = {"temperature": 0.8, "top_p": 0.8, "top_k": 5}
output = chat.invoke([HumanMessage(content="write a funny joke")], **kwargs)
logger.debug("output:", output)

"""
Or, run a stream call to get a stream response:
"""
logger.info("Or, run a stream call to get a stream response:")

outputs = chat.stream([HumanMessage(content="hi")], streaming=True)
for output in outputs:
    logger.debug("stream output:", output)

logger.info("\n\n[DONE]", bright=True)