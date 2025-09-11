from jet.logger import logger
from langchain_community.document_loaders.obs_file import OBSFileLoader
from obs import ObsClient
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
# Huawei OBS File
The following code demonstrates how to load an object from the Huawei OBS (Object Storage Service) as document.
"""
logger.info("# Huawei OBS File")




endpoint = "your-endpoint"


obs_client = ObsClient(
    access_key_id="your-access-key",
    secret_access_key="your-secret-key",
    server=endpoint,
)
loader = OBSFileLoader("your-bucket-name", "your-object-key", client=obs_client)

loader.load()

"""
## Each Loader with Separate Authentication Information
If you don't need to reuse OBS connections between different loaders, you can directly configure the `config`. The loader will use the config information to initialize its own OBS client.
"""
logger.info("## Each Loader with Separate Authentication Information")

config = {"ak": "your-access-key", "sk": "your-secret-key"}
loader = OBSFileLoader(
    "your-bucket-name", "your-object-key", endpoint=endpoint, config=config
)

loader.load()

"""
## Get Authentication Information from ECS
If your langchain is deployed on Huawei Cloud ECS and [Agency is set up](https://support.huaweicloud.com/intl/en-us/usermanual-ecs/ecs_03_0166.html#section7), the loader can directly get the security token from ECS without needing access key and secret key.
"""
logger.info("## Get Authentication Information from ECS")

config = {"get_token_from_ecs": True}
loader = OBSFileLoader(
    "your-bucket-name", "your-object-key", endpoint=endpoint, config=config
)

loader.load()

"""
## Access a Publicly Accessible Object
If the object you want to access allows anonymous user access (anonymous users have `GetObject` permission), you can directly load the object without configuring the `config` parameter.
"""
logger.info("## Access a Publicly Accessible Object")

loader = OBSFileLoader("your-bucket-name", "your-object-key", endpoint=endpoint)

loader.load()

logger.info("\n\n[DONE]", bright=True)