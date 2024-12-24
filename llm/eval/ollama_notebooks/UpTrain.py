```python
# %pip install -qU uptrain llama-index
import httpx
import os
import openai
import pandas as pd
from jet.llm.ollama import initialize_ollama_settings
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from uptrain import Evals, EvalLlamaIndex, Settings as UpTrainSettings
url = "https://uptrain-assets.s3.ap-south-1.amazonaws.com/data/nyc_text.txt"
if not os.path.exists("nyc_wikipedia"):
    os.makedirs("nyc_wikipedia")
dataset_path = os.path.join("./nyc_wikipedia", "nyc_text.txt")

if not os.path.exists(dataset_path):
    r = httpx.get(url)
    with open(dataset_path, "wb") as f:
        f.write(r.content)
data = [
    {"question": "What is the population of New York City?"},
    {"question": "What is the area of New York City?"},
    {"question": "What is the largest borough in New York City?"},
    {"question": "What is the average temperature in New York City?"},
    {"question": "What is the main airport in New York City?"},
    {"question": "What is the famous landmark in New York City?"},
    {"question": "What is the official language of New York City?"},
    {"question": "What is the currency used in New York City?"},
    {"question": "What is the time zone of New York City?"},
    {"question": "What is the famous sports team in New York City?"},
]
initialize_ollama_settings()
openai.api_key = "sk-************************"  # your OpenAI API key
Settings.chunk_size = 512

documents = SimpleDirectoryReader("./nyc_wikipedia/").load_data()

vector_index = VectorStoreIndex.from_documents(
    documents,
)

query_engine = vector_index.as_query_engine()
settings = UpTrainSettings(
    openai_api_key=openai.api_key,
)
llamaindex_object = EvalLlamaIndex(
    settings=settings, query_engine=query_engine
)
results = llamaindex_object.evaluate(
    project_name="uptrain-llama-index",
    evaluation_name="nyc_wikipedia",  # adding project and evaluation names allow you to track the results in the UpTrain dashboard
    data=data,
    checks=[Evals.CONTEXT_RELEVANCE, Evals.RESPONSE_CONCISENESS],
)
pd.DataFrame(results)
UPTRAIN_API_KEY = "up-**********************"  # your UpTrain API key

settings = UpTrainSettings(
    uptrain_access_token=UPTRAIN_API_KEY,
)
llamaindex_object = EvalLlamaIndex(
    settings=settings, query_engine=query_engine
)
results = llamaindex_object.evaluate(
    project_name="uptrain-llama-index",
    evaluation_name="nyc_wikipedia",  # adding project and evaluation names allow you to track the results in the UpTrain dashboard
    data=data,
    checks=[Evals.CONTEXT_RELEVANCE, Evals.RESPONSE_CONCISENESS],
)
pd.DataFrame(results)
```