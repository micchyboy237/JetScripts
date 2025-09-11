from jet.logger import logger
from langchain_community.agent_toolkits import CogniswitchToolkit
from langchain_community.tools.cogniswitch.tool import CogniswitchKnowledgeRequest
from langchain_community.tools.cogniswitch.tool import CogniswitchKnowledgeSourceFile
from langchain_community.tools.cogniswitch.tool import CogniswitchKnowledgeSourceURL
from langchain_community.tools.cogniswitch.tool import CogniswitchKnowledgeStatus
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
# CogniSwitch

>[CogniSwitch](https://www.cogniswitch.ai/aboutus) is an API based data platform that
> enhances enterprise data by extracting entities, concepts and their relationships
> thereby converting this data into a multidimensional format and storing it in
> a database that can accommodate these enhancements. In our case the data is stored
> in a knowledge graph. This enhanced data is now ready for consumption by LLMs and
> other GenAI applications ensuring the data is consumable and context can be maintained.
> Thereby eliminating hallucinations and delivering accuracy.

## Toolkit

See [installation instructions and usage example](/docs/integrations/tools/cogniswitch).
"""
logger.info("# CogniSwitch")


"""
## Tools

### CogniswitchKnowledgeRequest

>Tool that uses the CogniSwitch service to answer questions.
"""
logger.info("## Tools")


"""
### CogniswitchKnowledgeSourceFile

>Tool that uses the CogniSwitch services to store data from file.
"""
logger.info("### CogniswitchKnowledgeSourceFile")


"""
### CogniswitchKnowledgeSourceURL

>Tool that uses the CogniSwitch services to store data from a URL.
"""
logger.info("### CogniswitchKnowledgeSourceURL")


"""
### CogniswitchKnowledgeStatus

>Tool that uses the CogniSwitch services to get the status of the document or url uploaded.
"""
logger.info("### CogniswitchKnowledgeStatus")


logger.info("\n\n[DONE]", bright=True)