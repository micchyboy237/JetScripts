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
"""

"""
### Import Required Libraries
"""

import warnings

warnings.filterwarnings("ignore")
from llama_index.core.query_engine import CogniswitchQueryEngine

"""
### Cogniswitch Credentials and Ollama token
"""



"""
### Instantiate the Query Engine
"""

query_engine = CogniswitchQueryEngine(
    cs_token=cs_token, OAI_token=OAI_token, apiKey=oauth_token
)

"""
### Use the query_engine to chat with your knowledge
"""

answer_response = query_engine.query_knowledge("tell me about cogniswitch")
print(answer_response)