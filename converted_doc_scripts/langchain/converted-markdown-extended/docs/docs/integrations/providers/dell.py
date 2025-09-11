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
# Dell

Dell is a global technology company that provides a range of hardware, software, and
services, including AI solutions. Their AI portfolio includes purpose-built
infrastructure for AI workloads, including Dell PowerScale storage systems optimized
for AI data management.

## PowerScale

Dell [PowerScale](https://www.dell.com/en-us/shop/powerscale-family/sf/powerscale) is
an enterprise scale out storage system that hosts industry leading OneFS filesystem
that can be hosted on-prem or deployed in the cloud.

### Installation and Setup
"""
logger.info("# Dell")

pip install powerscale-rag-connector

"""
### Document loaders

See detail on available loaders [here](/docs/integrations/document_loaders/powerscale).
"""
logger.info("### Document loaders")

logger.info("\n\n[DONE]", bright=True)