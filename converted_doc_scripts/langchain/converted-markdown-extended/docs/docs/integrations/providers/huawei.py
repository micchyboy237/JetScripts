from jet.logger import logger
from langchain_community.document_loaders import OBSDirectoryLoader
from langchain_community.document_loaders.obs_file import OBSFileLoader
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
# Huawei

>[Huawei Technologies Co., Ltd.](https://www.huawei.com/) is a Chinese multinational
> digital communications technology corporation.
>
>[Huawei Cloud](https://www.huaweicloud.com/intl/en-us/product/) provides a comprehensive suite of
> global cloud computing services.


## Installation and Setup

To access the `Huawei Cloud`, you need an access token.

You also have to install a python library:
"""
logger.info("# Huawei")

pip install -U esdk-obs-python

"""
## Document Loader

### Huawei OBS Directory

See a [usage example](/docs/integrations/document_loaders/huawei_obs_directory).
"""
logger.info("## Document Loader")


"""
### Huawei OBS File

See a [usage example](/docs/integrations/document_loaders/huawei_obs_file).
"""
logger.info("### Huawei OBS File")


logger.info("\n\n[DONE]", bright=True)