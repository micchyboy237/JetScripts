from jet.llm.mlx.adapters.mlx_llama_index_llm_adapter import MLXLlamaIndexLLMAdapter
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.llms import ChatMessage
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.portkey import Portkey
import json
import os
import portkey as pk
import requests
import shutil
import time


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
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/portkey.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# Portkey

**Portkey** is a full-stack LLMOps platform that productionizes your Gen AI app reliably and securely.

#### Key Features of Portkey's Integration with Llamaindex:

<img src="https://3798672042-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FeWEp2XRBGxs7C1jgAdk7%2Fuploads%2FjDGBQvw5aFOCqctr0xwp%2FColab%20Version%202.png?alt=media&token=16057c99-b86c-416c-932e-c2b71549c506" alt="header" width=600 />


1. **[🚪 AI Gateway](#🔁-implementing-fallbacks-and-retries-with-portkey)**:
    - **[Automated Fallbacks & Retries](#🔁-implementing-fallbacks-and-retries-with-portkey)**: Ensure your application remains functional even if a primary service fails.
    - **[Load Balancing](#⚖️-implementing-load-balancing-with-portkey)**: Efficiently distribute incoming requests among multiple models.
    - **[Semantic Caching](#🧠-implementing-semantic-caching-with-portkey)**: Reduce costs and latency by intelligently caching results.
2. **[🔬 Observability](#🔬-observability-with-portkey)**:
    - **Logging**: Keep track of all requests for monitoring and debugging.
    - **Requests Tracing**: Understand the journey of each request for optimization.
    - **Custom Tags**: Segment and categorize requests for better insights.
3. **[📝 Continuous Improvement with User Feedback](#📝-feedback-with-portkey)**:
    - **Feedback Collection**: Seamlessly gather feedback on any served request, be it on a generation or conversation level.
    - **Weighted Feedback**: Obtain nuanced information by attaching weights to user feedback values.
    - **Feedback Metadata**: Incorporate custom metadata with the feedback to provide context, allowing for richer insights and analyses.
4. **[🔑 Secure Key Management](#feedback-with-portkey)**:
    - **Virtual Keys**: Portkey transforms original provider keys into virtual keys, ensuring your primary credentials remain untouched.
    - **Multiple Identifiers**: Ability to add multiple keys for the same provider or the same key under different names for easy identification without compromising security.

To harness these features, let's start with the setup:

<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/portkey.ipynb" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt=\"Open In Colab\" width=150 />
</a>

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# Portkey")

# %pip install llama-index-llms-portkey

# !pip install llama-index

# !pip install -U llama_index
# !pip install -U portkey-ai


"""
You do not need to install **any** other SDKs or import them in your Llamaindex app.

#### **Step 1️⃣: Get your Portkey API Key and your Virtual Keys for MLX, Anthropic, and more**

**[Portkey API Key](https://app.portkey.ai/)**: Log into [Portkey here](https://app.portkey.ai/), then click on the profile icon on top left and "Copy API Key".
"""
logger.info("#### **Step 1️⃣: Get your Portkey API Key and your Virtual Keys for MLX, Anthropic, and more**")


os.environ["PORTKEY_API_KEY"] = "PORTKEY_API_KEY"

"""
**[Virtual Keys](https://docs.portkey.ai/key-features/ai-provider-keys)**
1. Navigate to the "Virtual Keys" page on [Portkey dashboard](https://app.portkey.ai/) and hit the "Add Key" button located at the top right corner.
2. Choose your AI provider (MLX, Anthropic, Cohere, HuggingFace, etc.), assign a unique name to your key, and, if needed, jot down any relevant usage notes. Your virtual key is ready!

<img src="https://3798672042-files.gitbook.io/~/files/v0/b/gitbook-x-prod.appspot.com/o/spaces%2FeWEp2XRBGxs7C1jgAdk7%2Fuploads%2F66S1ik16Gle8jS1u6smr%2Fvirtual_keys.png?alt=media&token=2fec1c39-df4e-4c93-9549-7445a833321c" alt="header" width=600 />
3. Now copy and paste the keys below - you can use them anywhere within the Portkey ecosystem and keep your original key secure and untouched.
"""
logger.info("1. Navigate to the "Virtual Keys" page on [Portkey dashboard](https://app.portkey.ai/) and hit the "Add Key" button located at the top right corner.")

openai_virtual_key_a = ""
openai_virtual_key_b = ""

anthropic_virtual_key_a = ""
anthropic_virtual_key_b = ""

cohere_virtual_key_a = ""
cohere_virtual_key_b = ""

"""
If you don't want to use Portkey's Virtual keys, you can also use your AI provider keys directly.
"""
logger.info("If you don't want to use Portkey's Virtual keys, you can also use your AI provider keys directly.")

# os.environ["OPENAI_API_KEY"] = ""
# os.environ["ANTHROPIC_API_KEY"] = ""

"""
#### **Step 2️⃣: Configure Portkey Features**

To harness the full potential of Portkey's integration with Llamaindex, you can configure various features as illustrated above. Here's a guide to all Portkey features and the expected values:

| Feature             | Config Key              | Value(Type)                                      | Required    |
|---------------------|-------------------------|--------------------------------------------------|-------------|
| API Key             | `api_key`               | `string`                                         | ✅ Required (can be set externally) |
| Mode                | `mode`                  | `fallback`, `loadbalance`, `single`              | ✅ Required |
| Cache Type          | `cache_status`          | `simple`, `semantic`                             | ❔ Optional |
| Force Cache Refresh | `cache_force_refresh`   | `True`, `False`                                  | ❔ Optional |
| Cache Age           | `cache_age`             | `integer` (in seconds)                           | ❔ Optional |
| Trace ID            | `trace_id`              | `string`                                         | ❔ Optional |
| Retries         | `retry`           | `integer` [0,5]                                  | ❔ Optional |
| Metadata            | `metadata`              | `json object` [More info](https://docs.portkey.ai/key-features/custom-metadata)          | ❔ Optional |
| Base URL | `base_url` | `url` | ❔ Optional |

* `api_key` and `mode` are required values.
* You can set your Portkey API key using the Portkey constructor or you can also set it as an environment variable.
* There are **3** modes - Single, Fallback, Loadbalance.
  * **Single** - This is the standard mode. Use it if you do not want Fallback OR Loadbalance features.
  * **Fallback** - Set this mode if you want to enable the Fallback feature. [Check out the guide here](#implementing-fallbacks-and-retries-with-portkey).
  * **Loadbalance** - Set this mode if you want to enable the Loadbalance feature. [Check out the guide here](#implementing-load-balancing-with-portkey).

Here's an example of how to set up some of these features:
"""
logger.info("#### **Step 2️⃣: Configure Portkey Features**")

portkey_client = Portkey(
    mode="single",
)

"""
#### **Step 3️⃣: Constructing the LLM**

With the Portkey integration, constructing an LLM is simplified. Use the `LLMOptions` function for all providers, with the exact same keys you're accustomed to in your MLX or Anthropic constructors. The only new key is `weight`, essential for the load balancing feature.
"""
logger.info("#### **Step 3️⃣: Constructing the LLM**")

openai_llm = pk.LLMOptions(
    provider="openai",
    model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats",
    virtual_key=openai_virtual_key_a,
)

"""
The above code illustrates how to utilize the `LLMOptions` function to set up an LLM with the MLX provider and the GPT-4 model. This same function can be used for other providers as well, making the integration process streamlined and consistent across various providers.

#### **Step 4️⃣: Activate the Portkey Client**

Once you've constructed the LLM using the `LLMOptions` function, the next step is to activate it with Portkey. This step is essential to ensure that all the Portkey features are available for your LLM.
"""
logger.info("#### **Step 4️⃣: Activate the Portkey Client**")

portkey_client.add_llms(openai_llm)

"""
And, that's it! In just 4 steps, you have infused your Llamaindex app with sophisticated production capabilities.

#### **🔧 Testing the Integration**

Let's ensure that everything is set up correctly. Below, we create a simple chat scenario and pass it through our Portkey client to see the response.
"""
logger.info("#### **🔧 Testing the Integration**")

messages = [
    ChatMessage(role="system", content="You are a helpful assistant"),
    ChatMessage(role="user", content="What can you do?"),
]
logger.debug("Testing Portkey Llamaindex integration:")
response = portkey_client.chat(messages)
logger.debug(response)

"""
Here's how your logs will appear on your [Portkey dashboard](https://app.portkey.ai/):

<img src="https://portkey.ai/blog/content/images/2023/09/Log-1.png" alt="Logs" width=600 />

#### **⏩ Streaming Responses**

With Portkey, streaming responses has never been more straightforward. Portkey has 4 response functions:

1. `.complete(prompt)`
2. `.stream_complete(prompt)`
3. `.chat(messages)`
4. `.stream_chat(messages)`

While the `complete` function expects a string input(`str`), the `chat` function works with an array of `ChatMessage` objects.

**Example usage:**
"""
logger.info("#### **⏩ Streaming Responses**")

prompt = "Why is the sky blue?"

logger.debug("\nTesting Stream Complete:\n")
response = portkey_client.stream_complete(prompt)
for i in response:
    logger.debug(i.delta, end="", flush=True)


messages = [
    ChatMessage(role="system", content="You are a helpful assistant"),
    ChatMessage(role="user", content="What can you do?"),
]

logger.debug("\nTesting Stream Chat:\n")
response = portkey_client.stream_chat(messages)
for i in response:
    logger.debug(i.delta, end="", flush=True)

"""
#### **🔍 Recap and References**

Congratulations! 🎉 You've successfully set up and tested the Portkey integration with Llamaindex. To recap the steps:

1. pip install portkey-ai
2. from llama_index.llms import Portkey
3. Grab your Portkey API Key and create your virtual provider keys from [here](https://app.portkey.ai/).
4. Construct your Portkey client and set mode: `portkey_client=Portkey(mode="fallback")`
5. Construct your provider LLM with LLMOptions: `openai_llm = pk.LLMOptions(provider="openai", model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats", virtual_key=openai_key_a)`
6. Add the LLM to Portkey with `portkey_client.add_llms(openai_llm)`
7. Call the Portkey methods regularly like you would any other LLM, with `portkey_client.chat(messages)`

Here's the guide to all the functions and their params:
- [Portkey LLM Constructor](#step-2-add-all-the-portkey-features-you-want-as-illustrated-below-by-calling-the-portkey-class)
- [LLMOptions Constructor](https://github.com/Portkey-AI/rubeus-python-sdk/blob/4cf3e17b847225123e92f8e8467b41d082186d60/rubeus/api_resources/utils.py#L179)
- [List of Portkey + Llamaindex Features](#portkeys-integration-with-llamaindex-adds-the-following-production-capabilities-to-your-apps-out-of-the-box)

<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/llm/portkey.ipynb" target="_blank">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt=\"Open In Colab\" width=150 />
</a>

#### **🔁 Implementing Fallbacks and Retries with Portkey**

Fallbacks and retries are essential for building resilient AI applications. With Portkey, implementing these features is straightforward:

- **Fallbacks**: If a primary service or model fails, Portkey will automatically switch to a backup model.
- **Retries**: If a request fails, Portkey can be configured to retry the request multiple times.

Below, we demonstrate how to set up fallbacks and retries using Portkey:
"""
logger.info("#### **🔍 Recap and References**")

portkey_client = Portkey(mode="fallback")
messages = [
    ChatMessage(role="system", content="You are a helpful assistant"),
    ChatMessage(role="user", content="What can you do?"),
]

llm1 = pk.LLMOptions(
    provider="openai",
    model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats",
    retry_settings={"on_status_codes": [429, 500], "attempts": 2},
    virtual_key=openai_virtual_key_a,
)

llm2 = pk.LLMOptions(
    provider="openai",
    model="qwen3-0.6b-4bit", log_dir=f"{OUTPUT_DIR}/chats",
    virtual_key=openai_virtual_key_b,
)

portkey_client.add_llms(llm_params=[llm1, llm2])

logger.debug("Testing Fallback & Retry functionality:")
response = portkey_client.chat(messages)
logger.debug(response)

"""
#### **⚖️ Implementing Load Balancing with Portkey**

Load balancing ensures that incoming requests are efficiently distributed among multiple models. This not only enhances the performance but also provides redundancy in case one model fails.

With Portkey, implementing load balancing is simple. You need to:

- Define the `weight` parameter for each LLM. This weight determines how requests are distributed among the LLMs.
- Ensure that the sum of weights for all LLMs equals 1.

Here's an example of setting up load balancing with Portkey:
"""
logger.info("#### **⚖️ Implementing Load Balancing with Portkey**")

portkey_client = Portkey(mode="ab_test")

messages = [
    ChatMessage(role="system", content="You are a helpful assistant"),
    ChatMessage(role="user", content="What can you do?"),
]

llm1 = pk.LLMOptions(
    provider="openai",
    model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats",
    virtual_key=openai_virtual_key_a,
    weight=0.2,
)

llm2 = pk.LLMOptions(
    provider="openai",
    model="qwen3-0.6b-4bit", log_dir=f"{OUTPUT_DIR}/chats",
    virtual_key=openai_virtual_key_a,
    weight=0.8,
)

portkey_client.add_llms(llm_params=[llm1, llm2])

logger.debug("Testing Loadbalance functionality:")
response = portkey_client.chat(messages)
logger.debug(response)

"""
#### **🧠 Implementing Semantic Caching with Portkey**

Semantic caching is a smart caching mechanism that understands the context of a request. Instead of caching based solely on exact input matches, semantic caching identifies similar requests and serves cached results, reducing redundant requests and improving response times as well as saving money.

Let's see how to implement semantic caching with Portkey:
"""
logger.info("#### **🧠 Implementing Semantic Caching with Portkey**")


portkey_client = Portkey(mode="single")

openai_llm = pk.LLMOptions(
    provider="openai",
    model="qwen3-0.6b-4bit", log_dir=f"{OUTPUT_DIR}/chats",
    virtual_key=openai_virtual_key_a,
    cache_status="semantic",
)

portkey_client.add_llms(openai_llm)

current_messages = [
    ChatMessage(role="system", content="You are a helpful assistant"),
    ChatMessage(role="user", content="What are the ingredients of a pizza?"),
]

logger.debug("Testing Portkey Semantic Cache:")

start = time.time()
response = portkey_client.chat(current_messages)
end = time.time() - start

logger.debug(response)
logger.debug(f"{'-'*50}\nServed in {end} seconds.\n{'-'*50}")

new_messages = [
    ChatMessage(role="system", content="You are a helpful assistant"),
    ChatMessage(role="user", content="Ingredients of pizza"),
]

logger.debug("Testing Portkey Semantic Cache:")

start = time.time()
response = portkey_client.chat(new_messages)
end = time.time() - start

logger.debug(response)
logger.debug(f"{'-'*50}\nServed in {end} seconds.\n{'-'*50}")

"""
Portkey's cache supports two more cache-critical functions - Force Refresh and Age.

`cache_force_refresh`: Force-send a request to your provider instead of serving it from a cache.
`cache_age`: Decide the interval at which the cache store for this particular string should get automatically refreshed. The cache age is set in seconds.

Here's how you can use it:
"""
logger.info("Portkey's cache supports two more cache-critical functions - Force Refresh and Age.")

openai_llm = pk.LLMOptions(
    provider="openai",
    model="qwen3-0.6b-4bit", log_dir=f"{OUTPUT_DIR}/chats",
    virtual_key=openai_virtual_key_a,
    cache_force_refresh=True,
    cache_age=60,
)

"""
#### **🔬 Observability with Portkey**

Having insight into your application's behavior is paramount. Portkey's observability features allow you to monitor, debug, and optimize your AI applications with ease. You can track each request, understand its journey, and segment them based on custom tags. This level of detail can help in identifying bottlenecks, optimizing costs, and enhancing the overall user experience.

Here's how to set up observability with Portkey:
"""
logger.info("#### **🔬 Observability with Portkey**")

metadata = {
    "_environment": "production",
    "_prompt": "test",
    "_user": "user",
    "_organisation": "acme",
}

trace_id = "llamaindex_portkey"

portkey_client = Portkey(mode="single")

openai_llm = pk.LLMOptions(
    provider="openai",
    model="qwen3-0.6b-4bit", log_dir=f"{OUTPUT_DIR}/chats",
    virtual_key=openai_virtual_key_a,
    metadata=metadata,
    trace_id=trace_id,
)

portkey_client.add_llms(openai_llm)

logger.debug("Testing Observability functionality:")
response = portkey_client.chat(messages)
logger.debug(response)

"""
#### **🌉 Open Source AI Gateway**

Portkey's AI Gateway uses the [open source project Rubeus](https://github.com/portkey-ai/rubeus) internally. Rubeus powers features like interoperability of LLMs, load balancing, fallbacks, and acts as an intermediary, ensuring that your requests are processed optimally.

One of the advantages of using Portkey is its flexibility. You can easily customize its behavior, redirect requests to different providers, or even bypass logging to Portkey altogether.

Here's an example of customizing the behavior with Portkey:

```py
portkey_client.base_url=None
```

#### **📝 Feedback with Portkey**

Continuous improvement is a cornerstone of AI. To ensure your models and applications evolve and serve users better, feedback is vital. Portkey's Feedback API offers a straightforward way to gather weighted feedback from users, allowing you to refine and improve over time.

Here's how to utilize the Feedback API with Portkey:

Read more about [Feedback here](https://docs.portkey.ai/key-features/feedback-api).
"""
logger.info("#### **🌉 Open Source AI Gateway**")


url = "https://api.portkey.ai/v1/feedback"

headers = {
    "x-portkey-api-key": os.environ.get("PORTKEY_API_KEY"),
    "Content-Type": "application/json",
}

data = {"trace_id": "llamaindex_portkey", "value": 1}

response = requests.post(url, headers=headers, data=json.dumps(data))

logger.debug(response.text)

"""
All the feedback with `weight` and `value` for each trace id is available on the Portkey dashboard:

<img src="https://portkey.ai/blog/content/images/2023/09/feedback.png" alt="Feedback" width=600 />

#### **✅ Conclusion**

Integrating Portkey with Llamaindex simplifies the process of building robust and resilient AI applications. With features like semantic caching, observability, load balancing, feedback, and fallbacks, you can ensure optimal performance and continuous improvement.

By following this guide, you've set up and tested the Portkey integration with Llamaindex. As you continue to build and deploy AI applications, remember to leverage the full potential of this integration!

For further assistance or questions, reach out to the developers ➡️ <br />
<a href="https://twitter.com/intent/follow?screen_name=portkeyai" target="_blank">
  <img src="https://img.shields.io/twitter/follow/portkeyai?style=social&logo=twitter" alt="Twitter">
</a>

Join our community of practitioners putting LLMs into production ➡️ <br />
<a href="https://discord.gg/sDk9JaNfK8" target="_blank">
  <img src="https://img.shields.io/discord/1143393887742861333?logo=discord" alt="Discord">
</a>
"""
logger.info("#### **✅ Conclusion**")

logger.info("\n\n[DONE]", bright=True)