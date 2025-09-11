from jet.logger import logger
from langchain.chains import LLMChain
from langchain_community.llms.pai_eas_endpoint import PaiEasEndpoint
from langchain_core.prompts import PromptTemplate
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
# Alibaba Cloud PAI EAS

>[Machine Learning Platform for AI of Alibaba Cloud](https://www.alibabacloud.com/help/en/pai) is a machine learning or deep learning engineering platform intended for enterprises and developers. It provides easy-to-use, cost-effective, high-performance, and easy-to-scale plug-ins that can be applied to various industry scenarios. With over 140 built-in optimization algorithms, `Machine Learning Platform for AI` provides whole-process AI engineering capabilities including data labeling (`PAI-iTAG`), model building (`PAI-Designer` and `PAI-DSW`), model training (`PAI-DLC`), compilation optimization, and inference deployment (`PAI-EAS`). `PAI-EAS` supports different types of hardware resources, including CPUs and GPUs, and features high throughput and low latency. It allows you to deploy large-scale complex models with a few clicks and perform elastic scale-ins and scale-outs in real time. It also provides a comprehensive O&M and monitoring system.
"""
logger.info("# Alibaba Cloud PAI EAS")

# %pip install -qU langchain-community


template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate.from_template(template)

"""
One who wants to use EAS LLMs must set up EAS service first. When the EAS service is launched, `EAS_SERVICE_URL` and `EAS_SERVICE_TOKEN` can be obtained. Users can refer to https://www.alibabacloud.com/help/en/pai/user-guide/service-deployment/ for more information,
"""
logger.info("One who wants to use EAS LLMs must set up EAS service first. When the EAS service is launched, `EAS_SERVICE_URL` and `EAS_SERVICE_TOKEN` can be obtained. Users can refer to https://www.alibabacloud.com/help/en/pai/user-guide/service-deployment/ for more information,")


os.environ["EAS_SERVICE_URL"] = "Your_EAS_Service_URL"
os.environ["EAS_SERVICE_TOKEN"] = "Your_EAS_Service_Token"
llm = PaiEasEndpoint(
    eas_service_url=os.environ["EAS_SERVICE_URL"],
    eas_service_token=os.environ["EAS_SERVICE_TOKEN"],
)

llm_chain = prompt | llm

question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"
llm_chain.invoke({"question": question})

logger.info("\n\n[DONE]", bright=True)