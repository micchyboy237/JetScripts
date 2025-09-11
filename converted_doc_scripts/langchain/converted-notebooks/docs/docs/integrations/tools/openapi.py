from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.adapters.langchain.chat_ollama import Ollama
from jet.logger import logger
from langchain_community.agent_toolkits import OpenAPIToolkit, create_openapi_agent
from langchain_community.agent_toolkits.openapi import planner
from langchain_community.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain_community.tools.json.tool import JsonSpec
from langchain_community.utilities.requests import RequestsWrapper
import os
import shutil
import spotipy.util as util
import tiktoken
import yaml


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
# OpenAPI Toolkit

We can construct agents to consume arbitrary APIs, here APIs conformant to the `OpenAPI`/`Swagger` specification.
"""
logger.info("# OpenAPI Toolkit")

ALLOW_DANGEROUS_REQUEST = True

"""
## 1st example: hierarchical planning agent

In this example, we'll consider an approach called hierarchical planning, common in robotics and appearing in recent works for LLMs X robotics. We'll see it's a viable approach to start working with a massive API spec AND to assist with user queries that require multiple steps against the API.

The idea is simple: to get coherent agent behavior over long sequences behavior & to save on tokens, we'll separate concerns: a "planner" will be responsible for what endpoints to call and a "controller" will be responsible for how to call them.

In the initial implementation, the planner is an LLM chain that has the name and a short description for each endpoint in context. The controller is an LLM agent that is instantiated with documentation for only the endpoints for a particular plan. There's a lot left to get this working very robustly :)

---

### To start, let's collect some OpenAPI specs.
"""
logger.info("## 1st example: hierarchical planning agent")



"""
You will be able to get OpenAPI specs from here: [APIs-guru/openapi-directory](https://github.com/APIs-guru/openapi-directory)
"""
logger.info("You will be able to get OpenAPI specs from here: [APIs-guru/openapi-directory](https://github.com/APIs-guru/openapi-directory)")

# !wget https://raw.githubusercontent.com/ollama/ollama-openapi/master/openapi.yaml -O ollama_openapi.yaml
# !wget https://www.klarna.com/us/shopping/public/ollama/v0/api-docs -O klarna_openapi.yaml
# !wget https://raw.githubusercontent.com/APIs-guru/openapi-directory/main/APIs/spotify.com/1.0.0/openapi.yaml -O spotify_openapi.yaml


with open("ollama_openapi.yaml") as f:
    raw_ollama_api_spec = yaml.load(f, Loader=yaml.Loader)
ollama_api_spec = reduce_openapi_spec(raw_ollama_api_spec)

with open("klarna_openapi.yaml") as f:
    raw_klarna_api_spec = yaml.load(f, Loader=yaml.Loader)
klarna_api_spec = reduce_openapi_spec(raw_klarna_api_spec)

with open("spotify_openapi.yaml") as f:
    raw_spotify_api_spec = yaml.load(f, Loader=yaml.Loader)
spotify_api_spec = reduce_openapi_spec(raw_spotify_api_spec)

"""
---

We'll work with the Spotify API as one of the examples of a somewhat complex API. There's a bit of auth-related setup to do if you want to replicate this.

- You'll have to set up an application in the Spotify developer console, documented [here](https://developer.spotify.com/documentation/general/guides/authorization/), to get credentials: `CLIENT_ID`, `CLIENT_SECRET`, and `REDIRECT_URI`.
- To get an access tokens (and keep them fresh), you can implement the oauth flows, or you can use `spotipy`. If you've set your Spotify creedentials as environment variables `SPOTIPY_CLIENT_ID`, `SPOTIPY_CLIENT_SECRET`, and `SPOTIPY_REDIRECT_URI`, you can use the helper functions below:
"""
logger.info("We'll work with the Spotify API as one of the examples of a somewhat complex API. There's a bit of auth-related setup to do if you want to replicate this.")



def construct_spotify_auth_headers(raw_spec: dict):
    scopes = list(
        raw_spec["components"]["securitySchemes"]["oauth_2_0"]["flows"][
            "authorizationCode"
        ]["scopes"].keys()
    )
    access_token = util.prompt_for_user_token(scope=",".join(scopes))
    return {"Authorization": f"Bearer {access_token}"}


headers = construct_spotify_auth_headers(raw_spotify_api_spec)
requests_wrapper = RequestsWrapper(headers=headers)

"""
### How big is this spec?
"""
logger.info("### How big is this spec?")

endpoints = [
    (route, operation)
    for route, operations in raw_spotify_api_spec["paths"].items()
    for operation in operations
    if operation in ["get", "post"]
]
len(endpoints)


enc = tiktoken.encoding_for_model(model="llama3.2")


def count_tokens(s):
    return len(enc.encode(s))


count_tokens(yaml.dump(raw_spotify_api_spec))

"""
### Let's see some examples!

Starting with GPT-4. (Some robustness iterations under way for GPT-3 family.)
"""
logger.info("### Let's see some examples!")


llm = ChatOllama(model="llama3.2")

spotify_agent = planner.create_openapi_agent(
    spotify_api_spec,
    requests_wrapper,
    llm,
    allow_dangerous_requests=ALLOW_DANGEROUS_REQUEST,
)
user_query = (
    "make me a playlist with the first song from kind of blue. call it machine blues."
)
spotify_agent.invoke(user_query)

user_query = "give me a song I'd like, make it blues-ey"
spotify_agent.invoke(user_query)

"""
#### Try another API.
"""
logger.info("#### Try another API.")

# headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}
ollama_requests_wrapper = RequestsWrapper(headers=headers)

llm = ChatOllama(model="llama3.2")
ollama_agent = planner.create_openapi_agent(
    ollama_api_spec, ollama_requests_wrapper, llm
)
user_query = "generate a short piece of advice"
ollama_agent.invoke(user_query)

"""
Takes awhile to get there!

## 2nd example: "json explorer" agent

Here's an agent that's not particularly practical, but neat! The agent has access to 2 toolkits. One comprises tools to interact with json: one tool to list the keys of a json object and another tool to get the value for a given key. The other toolkit comprises `requests` wrappers to send GET and POST requests. This agent consumes a lot calls to the language model, but does a surprisingly decent job.
"""
logger.info("## 2nd example: "json explorer" agent")


with open("ollama_openapi.yaml") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
json_spec = JsonSpec(dict_=data, max_value_length=4000)


openapi_toolkit = OpenAPIToolkit.from_llm(
    Ollama(temperature=0), json_spec, ollama_requests_wrapper, verbose=True
)
openapi_agent_executor = create_openapi_agent(
    llm=Ollama(temperature=0),
    toolkit=openapi_toolkit,
    allow_dangerous_requests=ALLOW_DANGEROUS_REQUEST,
    verbose=True,
)

openapi_agent_executor.run(
    "Make a post request to ollama /completions. The prompt should be 'tell me a joke.'"
)

logger.info("\n\n[DONE]", bright=True)