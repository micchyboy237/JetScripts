from apify_haystack import ApifyDatasetFromActorCall
from haystack import Document
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OllamaFunctionCallingAdapterGenerator
from haystack.components.preprocessors import DocumentCleaner
from jet.logger import CustomLogger
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
# Analyze Your Instagram Comments’ Vibe with Apify and Haystack

Author: Jiri Spilka ([Apify](https://apify.com/jiri.spilka))  
Idea: Bilge Yücel ([deepset.ai](https://github.com/bilgeyucel))

Ever wondered if your Instagram posts are truly vibrating among your audience?
In this cookbook, we'll show you how to use the [Instagram Comment Scraper](https://apify.com/apify/instagram-comment-scraper) Actor to download comments from your instagram post and analyze them using a large language model. All performed within the Haystack ecosystem using the [apify-haystack](https://github.com/apify/apify-haystack/tree/main) integration.

We'll start by using the Actor to download the comments, clean the data with the [DocumentCleaner](https://docs.haystack.deepset.ai/docs/documentcleaner) and then use the [OllamaFunctionCallingAdapterGenerator](https://docs.haystack.deepset.ai/docs/openaigenerator) to discover the vibe of the Instagram posts.

# Install dependencies
"""
logger.info("# Analyze Your Instagram Comments’ Vibe with Apify and Haystack")

# !pip install apify-haystack==0.1.4 haystack-ai

"""
## Set up the API keys

You need to have an Apify account and obtain [APIFY_API_TOKEN](https://docs.apify.com/platform/integrations/api).

# You also need an OllamaFunctionCallingAdapter account and [OPENAI_API_KEY](https://platform.openai.com/docs/quickstart)
"""
logger.info("## Set up the API keys")

# from getpass import getpass

# os.environ["APIFY_API_TOKEN"] = getpass("Enter YOUR APIFY_API_TOKEN")
# os.environ["OPENAI_API_KEY"] = getpass("Enter YOUR OPENAI_API_KEY")

"""
## Use the Haystack Pipeline to Orchestrate Instagram Comments Scraper, Comments Cleanup, and Analysis Using LLM

Now, let's decide which post to analyze. We can start with these two posts that might reveal some interesting insights:

- `@tiffintech` on [How to easily keep up with tech?](https://www.instagram.com/p/C_a9jcRuJZZ/)
- `@kamaharishis` on [Affordable Care Act](https://www.instagram.com/p/C_RgBzogufK)

We'll download the comments using the Instagram Scraper Actor. But first, we need to understand the output format of the Actor.

The output is in the following format:
```json
[
  {
    "text": "You've just uncovered the goldmine for me 😍 but I still love your news and updates!",
    "timestamp": "2024-09-02T16:27:09.000Z",
    "ownerUsername": "codingmermaid.ai",
    "ownerProfilePicUrl": "....",
    "postUrl": "https://www.instagram.com/p/C_a9jcRuJZZ/"
  },
  {
    "text": "Will check it out🙌",
    "timestamp": "2024-09-02T16:29:28.000Z",
    "ownerUsername": "author.parijat",
    "postUrl": "https://www.instagram.com/p/C_a9jcRuJZZ/"
  }
]
```
We will convert this JSON to a Haystack Document using the `dataset_mapping_function` as follows
"""
logger.info("## Use the Haystack Pipeline to Orchestrate Instagram Comments Scraper, Comments Cleanup, and Analysis Using LLM")


def dataset_mapping_function(dataset_item: dict) -> Document:
    return Document(content=dataset_item.get("text"), meta={"ownerUsername": dataset_item.get("ownerUsername")})

"""
Once we understand the Actor output format and have the `dataset_mapping_function`, we can setup the Haystack component to enable interaction between the Haystack and Apify.

First, we need to provide `actor_id`, `dataset_mapping_function` along with input parameters `run_input`.

We can define the `run_input` in three ways:  
- i) when creating the `ApifyDatasetFromActorCall` class  
- ii) as arguments in a pipeline.  
- iii) as argumennts to the `run()` function when we calling `ApifyDatasetFromActorCall.run()`   
- iv) as a combination of `i)` and `ii)` as shown in this cookbook.

For a detailed description of the input parameters, visit the [Instagram Comments Scraper page](https://apify.com/apify/instagram-comment-scraper).

Let's setup the `ApifyDatasetFromActorCall`
"""
logger.info("Once we understand the Actor output format and have the `dataset_mapping_function`, we can setup the Haystack component to enable interaction between the Haystack and Apify.")


document_loader = ApifyDatasetFromActorCall(
    actor_id="apify/instagram-comment-scraper",
    run_input={"resultsLimit": 50},
    dataset_mapping_function=dataset_mapping_function,
)

"""
Next, we'll define a `prompt` for the LLM and connect all the components in the [Pipeline](https://docs.haystack.deepset.ai/docs/pipelines).
"""
logger.info("Next, we'll define a `prompt` for the LLM and connect all the components in the [Pipeline](https://docs.haystack.deepset.ai/docs/pipelines).")


prompt = """
Analyze these Instagram comments to determine if the post is generating positive energy, excitement,
or high engagement. Focus on sentiment, emotional tone, and engagement patterns to conclude if
the post is 'vibrating' with high energy. Be concise."

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Analysis:
"""

cleaner = DocumentCleaner(remove_empty_lines=True, remove_extra_whitespaces=True, remove_repeated_substrings=True)
prompt_builder = PromptBuilder(template=prompt)
generator = OllamaFunctionCallingAdapterGenerator(model="llama3.2")


pipe = Pipeline()
pipe.add_component("loader", document_loader)
pipe.add_component("cleaner", cleaner)
pipe.add_component("prompt_builder", prompt_builder)
pipe.add_component("llm", generator)
pipe.connect("loader", "cleaner")
pipe.connect("cleaner", "prompt_builder")
pipe.connect("prompt_builder", "llm")

"""
After that, we can run the pipeline. The execution and analysis will take approximately 30-60 seconds.
"""
logger.info("After that, we can run the pipeline. The execution and analysis will take approximately 30-60 seconds.")

url = "https://www.instagram.com/p/C_a9jcRuJZZ/"

res = pipe.run({"loader": {"run_input": {"directUrls": [url]}}})
res.get("llm", {}).get("replies", ["No response"])[0]

"""
Now, let's us run the same analysis. This time with the @kamalaharris post
"""
logger.info("Now, let's us run the same analysis. This time with the @kamalaharris post")

url = "https://www.instagram.com/p/C_RgBzogufK/"

res = pipe.run({"loader": {"run_input": {"directUrls": [url]}}})
res.get("llm", {}).get("replies", ["No response"])[0]

"""
The analysis shows that the first post about [How to easily keep up with tech?](https://www.instagram.com/p/C_a9jcRuJZZ/) is vibrating with high energy:

*The Instagram comments reveal a strong level of engagement and positive energy. Emojis like 😍, 😂, ❤️, 🙌, and 🔥 are frequently used, indicating excitement and enthusiasm. Commenters express gratitude, excitement, and appreciation for the content. The tone is overwhelmingly positive, supportive, and encouraging, with many users tagging others to share the content. Overall, this post is generating a vibrant and highly engaged response.*

However, the post by `@kamalaharris` on the [Affordable Care Act](https://www.instagram.com/p/C_RgBzogufK) is (not surprisingly) sparking a lot of controversy with negative comments.

*The comments on this post are generating negative energy but with high engagement. There's a strong focus on political opinions, particularly concerning insurance companies, the Affordable Care Act, Trump, and Biden. Many comments express frustration, criticism, and disagreement, with some users discussing party affiliations or support for specific politicians. There are also mentions of misinformation and conspiracy theories. Engagement is high, with numerous comment threads delving into various political issues. Overall, this post is vibrating with intense energy, driven by political opinions, disagreements, and active discussions.*

💡 You might receive slightly different results, as the comments may have changed since the last run
"""
logger.info("The analysis shows that the first post about [How to easily keep up with tech?](https://www.instagram.com/p/C_a9jcRuJZZ/) is vibrating with high energy:")

logger.info("\n\n[DONE]", bright=True)