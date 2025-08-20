from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.query_engine import CogniswitchQueryEngine
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
import shutil
import warnings


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
## CogniswitchQueryEngine

**Use CogniSwitch to build production ready applications that can consume, organize and retrieve knowledge flawlessly. Using the framework of your choice, in this case LlamaIndex, CogniSwitch helps alleviate the stress of decision making when it comes to choosing the right storage and retrieval formats. It also eradicates reliability issues and hallucinations when it comes to responses that are generated. Start interacting with your knowledge in 3 simple steps!**

Visit [https://www.cogniswitch.ai/developer](https://www.cogniswitch.ai/developer?utm_source=llamaindex&utm_medium=llamaindexbuild&utm_id=dev).<br>

**Registration:**
- Signup with your email and verify your registration
- You will get a mail with a platform token and oauth token for using the services.

**Upload Knowledge:**
- There are two ways to add your knowledge into Cogniswitch.
1. You can sign-in to Cogniswitch website and upload your document files or submit a url from the Document Upload page.<br>
2. You can use the CogniswitchToolSpec in llama-hub tools to add document or a url in Cogniswitch.<br> 

**CogniswitchQueryEngine:**<br>
- Instantiate the cogniswitchQueryEngine with the tokens and API keys.
- Use query_knowledge function in the Query Engine and input your query. <br>
- You will get the answer from your knowledge as the response. <br>

### Import Required Libraries
"""
logger.info("## CogniswitchQueryEngine")


warnings.filterwarnings("ignore")

"""
### Cogniswitch Credentials and MLX token
"""
logger.info("### Cogniswitch Credentials and MLX token")



"""
### Instantiate the Query Engine
"""
logger.info("### Instantiate the Query Engine")

query_engine = CogniswitchQueryEngine(
    cs_token=cs_token, OAI_token=OAI_token, apiKey=oauth_token
)

"""
### Use the query_engine to chat with your knowledge
"""
logger.info("### Use the query_engine to chat with your knowledge")

answer_response = query_engine.query_knowledge("tell me about cogniswitch")
logger.debug(answer_response)

logger.info("\n\n[DONE]", bright=True)