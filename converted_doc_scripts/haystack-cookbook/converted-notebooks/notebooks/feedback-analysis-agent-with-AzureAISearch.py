from azure.search.documents.indexes.models import (
SemanticConfiguration,
SemanticField,
SemanticPrioritizedFields,
SemanticSearch
)
from haystack import Pipeline
from haystack.components.converters import JSONConverter
from haystack.components.embedders import AzureOllamaFunctionCallingAdapterDocumentEmbedder
from haystack.components.embedders import AzureOllamaFunctionCallingAdapterTextEmbedder
from haystack.components.generators.chat import AzureOllamaFunctionCallingAdapterChatGenerator
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.tools import ToolInvoker
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import ByteStream
from haystack.dataclasses import ChatMessage
from haystack.tools import Tool
from haystack_integrations.components.retrievers.azure_ai_search import AzureAISearchHybridRetriever
from haystack_integrations.document_stores.azure_ai_search import AzureAISearchDocumentStore
from jet.logger import CustomLogger
from json import loads, dumps
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer
from typing import Dict, List
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")


"""
# Feedback Analysis Agent with Azure AI Search and Haystack
*by Amna Mubashar (Haystack), and Khye Wei (Azure AI Search)*

This notebook demonstrates how you can build indexing and querying pipelines using Azure AI Search-Haystack integration. Additionally, you'll develop an interactive feedback review agent leveraging Haystack Tools.

## Install the required dependencies
"""
logger.info("# Feedback Analysis Agent with Azure AI Search and Haystack")

# %pip install "haystack-ai>=2.13.0"
# %pip install "azure-ai-search-haystack
# !pip install jq
# !pip install nltk=="3.9.1"
# !pip install jsonschema
# !pip install kagglehub

"""
## Loading and Preparing the Dataset
We will use an open dataset consisting of approx. 28000 customer reviews for a clothing store. The dataset is available at [Shopper Sentiments](https://www.kaggle.com/datasets/nelgiriyewithana/shoppersentiments).

We will load the dataset and convert it into a JSON format that can be used by Haystack.
"""
logger.info("## Loading and Preparing the Dataset")

path = kagglehub.dataset_download("nelgiriyewithana/shoppersentiments")

# import getpass, os

# os.environ["AZURE_AI_SEARCH_API_KEY"] = getpass.getpass("Your AZURE_AI_SEARCH_API_KEY: ")
# os.environ["AZURE_AI_SEARCH_ENDPOINT"] = getpass.getpass("Your AZURE_AI_SEARCH_ENDPOINT: ")
# os.environ["AZURE_OPENAI_ENDPOINT"] = getpass.getpass("Your AZURE_OPENAI_ENDPOINT: ")
# os.environ["AZURE_OPENAI_API_KEY"] = getpass.getpass("Your AZURE_OPENAI_API_KEY: ")


path = "<Path to the CSV file>"

df = pd.read_csv(path, encoding='latin1', nrows=200) # We are using 200 rows for testing purposes

df.rename(columns={'review-label': 'rating'}, inplace=True)
df['year'] = pd.to_datetime(df['year'], format='%Y %H:%M:%S').dt.year

json_data = {"reviews": loads(df.to_json(orient="records"))}

"""
Once we have the JSON data, we can convert it into a Haystack Document format using the `JSONConverter` component. Its important to remove any documents with no content as they will not be indexed.
"""
logger.info("Once we have the JSON data, we can convert it into a Haystack Document format using the `JSONConverter` component. Its important to remove any documents with no content as they will not be indexed.")

converter = JSONConverter(
  jq_schema=".reviews[]", content_key="review", extra_meta_fields={"store_location", "date", "month", "year", "rating"}
)
source = ByteStream.from_string(dumps(json_data))

documents = converter.run(sources=[source])['documents']
documents = [doc for doc in documents if doc.content is not None] # remove documents with no content

"""
Remove any non-ASCII characters and any regex patterns that are not alphanumeric using the `DocumentCleaner` component.
"""
logger.info("Remove any non-ASCII characters and any regex patterns that are not alphanumeric using the `DocumentCleaner` component.")

cleaner = DocumentCleaner(ascii_only=True, remove_regex="i12i12i12")
cleaned_documents=cleaner.run(documents=documents)

"""
## Setting up Azure AI Search and Indexing Pipeline

We set up an indexing pipeline with `AzureAISearchDocumentStore` by following these steps:
1. Configure semantic search for the index
2. Initialize the document store with custom metadata fields and semantic search configuration
3. Create an indexing pipeline that:
   - Generates embeddings for the documents using `AzureOllamaFunctionCallingAdapterDocumentEmbedder`
   - Writes the documents and their embeddings to the search index

The semantic configuration allows for more intelligent searching beyond simple keyword matching. Note, the metadata fields need to be declared while creating the index as the API does not allow modifying them after index creation.
"""
logger.info("## Setting up Azure AI Search and Indexing Pipeline")




semantic_config = SemanticConfiguration(
    name="my-semantic-config",
    prioritized_fields=SemanticPrioritizedFields(
        content_fields=[SemanticField(field_name="content")]
    )
)

semantic_search = SemanticSearch(configurations=[semantic_config])

document_store = AzureAISearchDocumentStore(index_name="customer-reviews-analysis",
    embedding_dimension=1536, metadata_fields = {"month": int, "year": int, "rating": int, "store_location": str}, semantic_search=semantic_search)

indexing_pipeline = Pipeline()
indexing_pipeline.add_component(instance=AzureOllamaFunctionCallingAdapterDocumentEmbedder(), name="document_embedder")
indexing_pipeline.add_component(instance=DocumentWriter(document_store=document_store), name="doc_writer")
indexing_pipeline.connect("document_embedder", "doc_writer")

indexing_pipeline.run({"document_embedder": {"documents": cleaned_documents["documents"]}})

"""
## Creating the Query Pipeline

Here we set up the query pipeline that will retrieve relevant reviews based on user queries. The pipeline consists of:

1. A text embedder (`AzureOllamaFunctionCallingAdapterTextEmbedder`) that converts user queries into embeddings.
2. A hybrid retriever (`AzureAISearchHybridRetriever`) that uses vector and semantic search to retrieve the most relevant reviews.
"""
logger.info("## Creating the Query Pipeline")



query_pipeline = Pipeline()
query_pipeline.add_component("text_embedder", AzureOllamaFunctionCallingAdapterTextEmbedder())
query_pipeline.add_component("retriever", AzureAISearchHybridRetriever(document_store=document_store, query_type="semantic", semantic_configuration_name="my-semantic-config", top_k=10))
query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")

query = "Which reviews are about shipping?"

result = query_pipeline.run({"text_embedder": {"text": query}, "retriever": {"query": query}})
retrieved_reviews = result["retriever"]["documents"]
logger.debug(retrieved_reviews)

"""
## Create Tools for Sentiment Analysis and Summarization
Install the required dependencies.
"""
logger.info("## Create Tools for Sentiment Analysis and Summarization")

# !pip install vaderSentiment
# !pip install matplotlib
# !pip install sumy

"""
Create a function that will be used by `review_analysis` tool to visualize the sentiment distribution across customer review aspects (e.g., product quality, shipping). It compares VADER-based sentiment scores with customer ratings using color-coded bars (positive, neutral, negative).
"""
logger.info("Create a function that will be used by `review_analysis` tool to visualize the sentiment distribution across customer review aspects (e.g., product quality, shipping). It compares VADER-based sentiment scores with customer ratings using color-coded bars (positive, neutral, negative).")



def plot_sentiment_distribution(aspects):
    data = [(topic, review['sentiment']['analyzer_rating'],
             review['review']['rating'], review['sentiment']['label'])
            for topic, reviews in aspects.items()
            for review in reviews]

    df = pd.DataFrame(data, columns=['Topic', 'Normalized Score', 'Original Rating', 'Sentiment'])

    df_means = df.groupby('Topic').agg({
        'Normalized Score': 'mean',
        'Original Rating': 'mean'
    }).reset_index()

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(df_means))
    bar_width = 0.3

    colors = {
        'positive': '#2ecc71',
        'neutral': '#f1c40f',
        'negative': '#e74c3c'
    }

    sentiment_colors = [colors[df.groupby('Topic')['Sentiment'].agg(lambda x: x.mode()[0])[topic]]
                       for topic in df_means['Topic']]

    bars1 = ax.bar(x - bar_width/2, df_means['Normalized Score'],
                   bar_width, label='Normalized Score', color=sentiment_colors)
    bars2 = ax.bar(x + bar_width/2, df_means['Original Rating'],
                   bar_width, label='Original Rating', color='gray', alpha=0.7)

    ax.set_ylabel('Score', fontsize=9)
    ax.set_title('Average Sentiment Scores by Topic', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(df_means['Topic'], rotation=45, ha='right', fontsize=8)
    ax.tick_params(axis='y', labelsize=8)

    for bars in [bars1, bars2]:
        ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=8)

    ax.legend(handles=[plt.Rectangle((0,0),1,1, color=c) for c in colors.values()] +
             [plt.Rectangle((0,0),1,1, color='gray', alpha=0.7)],
             labels=list(colors.keys()) + ['Original Rating'],
             loc='upper right',
             fontsize=8)

    plt.tight_layout()
    plt.show()

"""
Create a tool to perform aspect-based sentiment analysis on customer reviews using the VADER sentiment analyzer. It involves:

- Identifying specific aspects within reviews (e.g., product quality, shipping, customer service, pricing) using predefined keywords
- Calculating sentiment scores for each review mentioning these aspects
- Categorizing sentiment as 'positive', 'negative', or 'neutral' 
- Normalizing sentiment scores to a scale of 1 to 5 for comparison with customer ratings
"""
logger.info("Create a tool to perform aspect-based sentiment analysis on customer reviews using the VADER sentiment analyzer. It involves:")




def analyze_sentiment(reviews: List[Dict]) -> Dict:
    """
    Perform aspect-based sentiment analysis.

    For each review that mentions keywords related to a specific topic, the function computes
    sentiment scores using VADER and categorizes the sentiment as 'positive', 'negative', or 'neutral'.

    """
    aspects = {
        "product_quality": [],
        "shipping": [],
        "customer_service": [],
        "pricing": []
    }

    keywords = {
        "product_quality": ["quality", "material", "design", "fit", "size", "color", "style"],
        "shipping": ["shipping", "delivery", "arrived"],
        "customer_service": ["service", "support", "help"],
        "pricing": ["price", "cost", "expensive", "cheap"]
    }


    analyzer = SentimentIntensityAnalyzer()

    for review in reviews:
        text = review.get("review", "").lower()
        for topic, words in keywords.items():
            if any(word in text for word in words):
                sentiment_scores = analyzer.polarity_scores(text)

                compound = sentiment_scores['compound']
                normalized_score = (compound + 1) * 2 + 1

                if compound >= 0.03:
                    sentiment_label = 'positive'
                elif compound <= -0.03:
                    sentiment_label = 'negative'
                else:
                    sentiment_label = 'neutral'

                aspects[topic].append({
                    "review": review,
                    "sentiment": {
                        "analyzer_rating": normalized_score,
                        "label": sentiment_label
                    }
                })
    plot_sentiment_distribution(aspects)

    return {
        "total_reviews": len(reviews),
        "sentiment_analysis": aspects,
        "average_rating": sum(r.get("rating", 3) for r in reviews) / len(reviews)
    }

sentiment_tool = Tool(
    name="review_analysis",
    description="Aspect based sentiment analysis tool that compares the sentiment of reviews by analyzer and rating",
    function=analyze_sentiment,
    parameters={
        "type": "object",
        "properties": {
            "reviews": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "review": {"type": "string"},
                        "rating": {"type": "integer"},
                        "date": {"type": "string"}
                    }
                }
            },
        },
        "required": ["reviews"]
    }
)

"""
Create a tool for summarizing customer reviews. The process involves:

- Using the LSA (Latent Semantic Analysis) summarizer to identify and extract the most important sentences from each review
- Creating concise summaries that capture the essence of the reviews
"""
logger.info("Create a tool for summarizing customer reviews. The process involves:")



def summarize_reviews(reviews: List[Dict]) -> Dict:
    """
    Summarize the reviews by extracting key sentences.
    """
    summaries = []
    summarizer = LsaSummarizer()
    for review in reviews:
        text = review.get("review", "")
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summary = summarizer(parser.document, 2)  # Adjust the number of sentences as needed
        summary_text = " ".join(str(sentence) for sentence in summary)
        summaries.append({"review": text, "summary": summary_text})

    return {"summaries": summaries}

summarization_tool = Tool(
    name="review_summarization",
    description="Tool to summarize customer reviews by extracting key sentences.",
    function=summarize_reviews,
    parameters={
        "type": "object",
        "properties": {
            "reviews": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "review": {"type": "string"},
                        "rating": {"type": "integer"},
                        "date": {"type": "string"}
                    }
                }
            },
        },
        "required": ["reviews"]
    }
)

"""
## Creating an Interactive Feedback Review Agent

We now have the tools to build an interactive agent for customer feedback analysis. The agent dynamically selects the appropriate tool based on user queries, gathers insights based on tool response. The agent then uses the `AzureOllamaFunctionCallingAdapterChatGenerator` to combine the query, retrieved reviews, and tool responses into a comprehensive review analysis.
"""
logger.info("## Creating an Interactive Feedback Review Agent")


def create_review_agent():
    """Creates an interactive review analysis agent"""

    chat_generator = AzureOllamaFunctionCallingAdapterChatGenerator(
        tools=[sentiment_tool, summarization_tool]
    )

    system_message = ChatMessage.from_system(
        """
        You are a customer review analysis expert. Your task is to perform aspect based sentiment analysis on customer reviews.
        You can use two tools to get insights:
        - review_analysis: to get the sentiment of reviews by analyzer and rating
        - review_summarization: to get the summary of reviews.

        Depending on the user's question, use the appropriate tool to get insights and explain them in a helpful way.

        """
    )

    return chat_generator, system_message

tool_invoker = ToolInvoker(tools=[sentiment_tool, summarization_tool])

"""
Let's put our agent to the test with a sample query and see it in action! 🚀
"""
logger.info("Let's put our agent to the test with a sample query and see it in action! 🚀")

chat_generator, system_message = create_review_agent()

messages = [system_message]

while True:
    user_input = input("\n\nwaiting for input (type 'exit' or 'quit' to stop)\n: ")
    if user_input.lower() == "exit" or user_input.lower() == "quit":
        break
    messages.append(ChatMessage.from_user(user_input))

    print (f"\n🧑: {user_input}")
    user_prompt = ChatMessage.from_user(f"""
    {user_input}
    Here are the reviews with analysis:
    {retrieved_reviews}
    """)
    messages.append(user_prompt)

    while True:
        logger.debug("⌛ iterating...")

        replies = chat_generator.run(messages=messages)["replies"]
        messages.extend(replies)

        if not replies[0].tool_calls:
            break
        tool_calls = replies[0].tool_calls

        for tc in tool_calls:
            logger.debug("\n TOOL CALL:")
            logger.debug(f"\t{tc.tool_name}")

        tool_messages = tool_invoker.run(messages=replies)["tool_messages"]
        messages.extend(tool_messages)

    logger.debug(f"🤖: {messages[-1].text}")

logger.info("\n\n[DONE]", bright=True)