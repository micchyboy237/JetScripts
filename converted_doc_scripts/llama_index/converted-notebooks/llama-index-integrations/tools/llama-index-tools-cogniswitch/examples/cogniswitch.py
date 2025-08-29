from jet.logger import CustomLogger
from llama_index.agent import ReActAgent
from llama_index.tools.cogniswitch import CogniswitchToolSpec
import os
import shutil
import warnings


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
## Cogniswitch ToolSpec

**Use CogniSwitch to build production ready applications that can consume, organize and retrieve knowledge flawlessly. Using the framework of your choice, in this case LlamaIndex, CogniSwitch helps alleviate the stress of decision making when it comes to, choosing the right storage and retrieval formats. It also eradicates reliability issues and hallucinations when it comes to responses that are generated. Get started by interacting with your knowledge in a few simple steps**

visit [https://www.cogniswitch.ai/developer](https://www.cogniswitch.ai/developer?utm_source=llamaindex&utm_medium=llamaindexbuild&utm_id=dev).<br>

**Registration:**
- Signup with your email and verify your registration
- You will get a mail with a platform token and oauth token for using the services.


**Step 1: Instantiate the Cogniswitch ToolSpec:**<br>
- Use your cogniswitch token, openAI API key, oauth token to instantiate the toolspec. <br> 

**Step 2: Instantiate the Agent:**<br>
- Instantiate the agent with the list of tools from the toolspec. <br> 

**Step 3: Cogniswitch Store data:**<br>
- Make the call to the agent by giving the file path or url to the agent input. <br>
- The agent will pick the tool and use the file/url and it will be processed and stored in your knowledge store. <br> 
- You can check the status of document processing with a call to the agent. Alternatively you can also check in [cogniswitch console](- You can check the status of document processing with a call to the agent. Alternatively you can also check in [cogniswitch console](https://console.cogniswitch.ai:8443/login?utm_source=llamaindex&utm_medium=llamaindexbuild&utm_id=dev).<br>

**Step 4: Cogniswitch Answer:**<br>
- Make the call to the agent by giving query as agent input. <br>
- You will get the answer from your knowledge as the response. <br>

### Import Required Libraries
"""
logger.info("## Cogniswitch ToolSpec")


warnings.filterwarnings("ignore")



"""
### Cogniswitch Credentials and OllamaFunctionCallingAdapter token
"""
logger.info("### Cogniswitch Credentials and OllamaFunctionCallingAdapter token")



"""
### Instantiate the Tool Spec
"""
logger.info("### Instantiate the Tool Spec")

toolspec = CogniswitchToolSpec(cs_token=cs_token, apiKey=oauth_token)

"""
### Get the list of tools in the toolspec
"""
logger.info("### Get the list of tools in the toolspec")

tool_lst = toolspec.to_tool_list()

"""
### Instantiate the agent
"""
logger.info("### Instantiate the agent")

agent = ReActAgent.from_tools(tool_lst)

"""
### Use the agent for storing data in cogniswitch with a single call
"""
logger.info("### Use the agent for storing data in cogniswitch with a single call")

store_response = agent.chat("Upload this URL- https://cogniswitch.ai/developer")

logger.debug(store_response)

"""
### Use the agent for storing data from a file
"""
logger.info("### Use the agent for storing data from a file")

store_response = agent.chat("Upload this file- sample_file.txt")

logger.debug(store_response)

"""
### Use the agent to know the document status with a single call
"""
logger.info("### Use the agent to know the document status with a single call")

response = agent.chat("Tell me the status of Cogniswitch Developer Website")

logger.debug(response)

logger.debug(response.sources[0])

"""
### Use agent for answering with a single call
"""
logger.info("### Use agent for answering with a single call")

answer_response = agent.chat("How does cogniswitch help developers")

logger.debug(answer_response)

logger.info("\n\n[DONE]", bright=True)