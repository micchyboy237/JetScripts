from haystack import Document, Pipeline
from haystack.components.builders import AnswerBuilder
from haystack.components.builders import ChatPromptBuilder
from haystack.components.embedders import OllamaFunctionCallingAdapterTextEmbedder, OllamaFunctionCallingAdapterDocumentEmbedder
from haystack.components.generators import OllamaFunctionCallingAdapterGenerator
from haystack.components.generators.chat import OllamaFunctionCallingAdapterChatGenerator
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack_integrations.components.evaluators.ragas import RagasEvaluator
from jet.logger import CustomLogger
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset
from ragas.llms import HaystackLLMWrapper
from ragas.metrics import AnswerRelevancy, ContextPrecision, Faithfulness
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
# RAG pipeline evaluation using Ragas

[Ragas](https://docs.ragas.io/en/stable/) is an open source framework for model-based evaluation to evaluate your [Retrieval Augmented Generation](https://www.deepset.ai/blog/llms-retrieval-augmentation) (RAG) pipelines and LLM applications.
It supports metrics like correctness, tone, hallucination (faithfulness), fluency, and more.

For more information about evaluators, supported metrics and usage, check out:

* [RagasEvaluator](https://docs.haystack.deepset.ai/docs/ragasevaluator)
* [Model based evaluation](https://docs.haystack.deepset.ai/docs/model-based-evaluation)

This notebook shows how to use the [Ragas-Haystack](https://haystack.deepset.ai/integrations/ragas) integration to evaluate a RAG pipeline against various metrics.

Notebook by [*Anushree Bannadabhavi*](https://github.com/AnushreeBannadabhavi), [*Siddharth Sahu*](https://github.com/sahusiddharth), [*Julian Risch*](https://github.com/julian-risch)

## Prerequisites:

- **Ragas** uses [OllamaFunctionCallingAdapter](https://openai.com/) key for computing some metrics, so we need an OllamaFunctionCallingAdapter API key.
"""
logger.info("# RAG pipeline evaluation using Ragas")

# from getpass import getpass

# os.environ["OPENAI_API_KEY"] = getpass("Enter OllamaFunctionCallingAdapter API key:")

"""
## Install dependencies
"""
logger.info("## Install dependencies")

# !pip install ragas-haystack

"""
#### Importing Required Libraries
"""
logger.info("#### Importing Required Libraries")



"""
#### Creating a Sample Dataset
In this section we create a sample dataset containing information about AI companies and their language models. This dataset serves as the context for retrieving relevant data during pipeline execution.
"""
logger.info("#### Creating a Sample Dataset")

dataset = [
    "OllamaFunctionCallingAdapter is one of the most recognized names in the large language model space, known for its GPT series of models. These models excel at generating human-like text and performing tasks like creative writing, answering questions, and summarizing content. GPT-4, their latest release, has set benchmarks in understanding context and delivering detailed responses.",
    "OllamaFunctionCallingAdapter is well-known for its Claude series of language models, designed with a strong focus on safety and ethical AI behavior. Claude is particularly praised for its ability to follow complex instructions and generate text that aligns closely with user intent.",
    "DeepMind, a division of Google, is recognized for its cutting-edge Gemini models, which are integrated into various Google products like Bard and Workspace tools. These models are renowned for their conversational abilities and their capacity to handle complex, multi-turn dialogues.",
    "Meta AI is best known for its LLaMA (Large Language Model Meta AI) series, which has been made open-source for researchers and developers. LLaMA models are praised for their ability to support innovation and experimentation due to their accessibility and strong performance.",
    "Meta AI with it's LLaMA models aims to democratize AI development by making high-quality models available for free, fostering collaboration across industries. Their open-source approach has been a game-changer for researchers without access to expensive resources.",
    "Microsoft’s Azure AI platform is famous for integrating OllamaFunctionCallingAdapter’s GPT models, enabling businesses to use these advanced models in a scalable and secure cloud environment. Azure AI powers applications like Copilot in Office 365, helping users draft emails, generate summaries, and more.",
    "Amazon’s Bedrock platform is recognized for providing access to various language models, including its own models and third-party ones like OllamaFunctionCallingAdapter’s Claude and AI21’s Jurassic. Bedrock is especially valued for its flexibility, allowing users to choose models based on their specific needs.",
    "Cohere is well-known for its language models tailored for business use, excelling in tasks like search, summarization, and customer support. Their models are recognized for being efficient, cost-effective, and easy to integrate into workflows.",
    "AI21 Labs is famous for its Jurassic series of language models, which are highly versatile and capable of handling tasks like content creation and code generation. The Jurassic models stand out for their natural language understanding and ability to generate detailed and coherent responses.",
    "In the rapidly advancing field of artificial intelligence, several companies have made significant contributions with their large language models. Notable players include OllamaFunctionCallingAdapter, known for its GPT Series (including GPT-4); OllamaFunctionCallingAdapter, which offers the Claude Series; Google DeepMind with its Gemini Models; Meta AI, recognized for its LLaMA Series; Microsoft Azure AI, which integrates OllamaFunctionCallingAdapter’s GPT Models; Amazon AWS (Bedrock), providing access to various models including Claude (OllamaFunctionCallingAdapter) and Jurassic (AI21 Labs); Cohere, which offers its own models tailored for business use; and AI21 Labs, known for its Jurassic Series. These companies are shaping the landscape of AI by providing powerful models with diverse capabilities.",
]

"""
#### Initializing RAG Pipeline Components
This section sets up the essential components required to build a Retrieval-Augmented Generation (RAG) pipeline. These components include a Document Store for managing and storing documents, an Embedder for generating embeddings to enable similarity-based retrieval, and a Retriever for fetching relevant documents. Additionally, a Prompt Template is designed to structure the pipeline's input, while a Chat Generator handles response generation.
"""
logger.info("#### Initializing RAG Pipeline Components")

document_store = InMemoryDocumentStore()
docs = [Document(content=doc) for doc in dataset]

document_embedder = OllamaFunctionCallingAdapterDocumentEmbedder(model="mxbai-embed-large")
text_embedder = OllamaFunctionCallingAdapterTextEmbedder(model="mxbai-embed-large")

docs_with_embeddings = document_embedder.run(docs)
document_store.write_documents(docs_with_embeddings["documents"])

retriever = InMemoryEmbeddingRetriever(document_store, top_k=2)

template = [
    ChatMessage.from_user(
        """
Given the following information, answer the question.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
"""
    )
]

prompt_builder = ChatPromptBuilder(template=template, required_variables="*")
chat_generator = OllamaFunctionCallingAdapterChatGenerator(model="llama3.2")

"""
#### Configuring RagasEvaluator Component

Pass all the Ragas metrics you want to use for evaluation, ensuring that all the necessary information to calculate each selected metric is provided.

For example:

- **AnswerRelevancy**: requires both the **query** and the **response**. It does not consider factuality but instead assigns lower score to cases where the response lacks completeness or contains redundant details.
- **ContextPrecision**: requires the **query**, **retrieved documents**, and the **reference**. It evaluates to what extent the retrieved documents contain precisely only what is relevant to answer the query.
- **Faithfulness**: requires the **query**, **retrieved documents**, and the **response**. The response is regarded as faithful if all the claims that are made in the response can be inferred from the retrieved documents.

Make sure to include all relevant data for each metric to ensure accurate evaluation.
"""
logger.info("#### Configuring RagasEvaluator Component")

llm = OllamaFunctionCallingAdapterGenerator(model="llama3.2")
evaluator_llm = HaystackLLMWrapper(llm)

ragas_evaluator = RagasEvaluator(
    ragas_metrics=[AnswerRelevancy(), ContextPrecision(), Faithfulness()],
    evaluator_llm=evaluator_llm,
)

"""
#### Building and Connecting the RAG Pipeline
Here we add and connect the initialized components to form a RAG Haystack pipeline.
"""
logger.info("#### Building and Connecting the RAG Pipeline")

rag_pipeline = Pipeline()

rag_pipeline.add_component("text_embedder", text_embedder)
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", chat_generator)
rag_pipeline.add_component("answer_builder", AnswerBuilder())
rag_pipeline.add_component("ragas_evaluator", ragas_evaluator)

rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
rag_pipeline.connect("retriever", "prompt_builder")
rag_pipeline.connect("prompt_builder.prompt", "llm.messages")
rag_pipeline.connect("llm.replies", "answer_builder.replies")
rag_pipeline.connect("retriever", "answer_builder.documents")
rag_pipeline.connect("retriever", "ragas_evaluator.documents")
rag_pipeline.connect("llm.replies", "ragas_evaluator.response")

question = "What makes Meta AI’s LLaMA models stand out?"

reference = "Meta AI’s LLaMA models stand out for being open-source, supporting innovation and experimentation due to their accessibility and strong performance."


result = rag_pipeline.run(
    {
        "text_embedder": {"text": question},
        "prompt_builder": {"question": question},
        "answer_builder": {"query": question},
        "ragas_evaluator": {"query": question, "reference": reference},
    }
)

logger.debug(result['answer_builder']['answers'][0].data, '\n')
logger.debug(result['ragas_evaluator']['result'])

"""
## Standalone Evaluation of the RAG Pipeline

This section explores an alternative approach to evaluating a RAG pipeline without using the `RagasEvaluator` component. It emphasizes manual extraction of outputs and organizing them for evaluation.

You can use any existing Haystack pipeline for this purpose. For demonstration, we will create a simple RAG pipeline similar to the one described earlier, but without including the `RagasEvaluator` component.

#### Setting Up a Basic RAG Pipeline
We construct a simple RAG pipeline similar to the approach above but without the RagasEvaluator component.
"""
logger.info("## Standalone Evaluation of the RAG Pipeline")

document_store = InMemoryDocumentStore()
docs = [Document(content=doc) for doc in dataset]

document_embedder = OllamaFunctionCallingAdapterDocumentEmbedder(model="mxbai-embed-large")
text_embedder = OllamaFunctionCallingAdapterTextEmbedder(model="mxbai-embed-large")

docs_with_embeddings = document_embedder.run(docs)
document_store.write_documents(docs_with_embeddings["documents"])

retriever = InMemoryEmbeddingRetriever(document_store, top_k=2)

template = [
    ChatMessage.from_user(
        """
Given the following information, answer the question.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
"""
    )
]

prompt_builder = ChatPromptBuilder(template=template, required_variables="*")
chat_generator = OllamaFunctionCallingAdapterChatGenerator(model="llama3.2")

rag_pipeline = Pipeline()

rag_pipeline.add_component("text_embedder", text_embedder)
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", chat_generator)
rag_pipeline.add_component("answer_builder", AnswerBuilder())

rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
rag_pipeline.connect("retriever", "prompt_builder")
rag_pipeline.connect("prompt_builder.prompt", "llm.messages")
rag_pipeline.connect("llm.replies", "answer_builder.replies")
rag_pipeline.connect("retriever", "answer_builder.documents")
rag_pipeline.connect("llm.replies", "answer_builder.replies")
rag_pipeline.connect("retriever", "answer_builder.documents")

"""
#### Extracting Outputs for Evaluation
After building the pipeline, we use it to generate the necessary outputs, such as retrieved documents and responses. These outputs are then structured into a dataset for evaluation.
"""
logger.info("#### Extracting Outputs for Evaluation")

questions = [
    "Who are the major players in the large language model space?",
    "What is Microsoft’s Azure AI platform known for?",
    "What kind of models does Cohere provide?",
]

references = [
    "The major players include OllamaFunctionCallingAdapter (GPT Series), OllamaFunctionCallingAdapter (Claude Series), Google DeepMind (Gemini Models), Meta AI (LLaMA Series), Microsoft Azure AI (integrating GPT Models), Amazon AWS (Bedrock with Claude and Jurassic), Cohere (business-focused models), and AI21 Labs (Jurassic Series).",
    "Microsoft’s Azure AI platform is known for integrating OllamaFunctionCallingAdapter’s GPT models, enabling businesses to use these models in a scalable and secure cloud environment.",
    "Cohere provides language models tailored for business use, excelling in tasks like search, summarization, and customer support.",
]


evals_list = []

for que_idx in range(len(questions)):

    single_turn = {}
    single_turn['user_input'] = questions[que_idx]
    single_turn['reference'] = references[que_idx]

    response = rag_pipeline.run(
        {
            "text_embedder": {"text": questions[que_idx]},
            "prompt_builder": {"question": questions[que_idx]},
            "answer_builder": {"query": questions[que_idx]},
        }
    )

    single_turn['response'] = response["answer_builder"]["answers"][0].data

    haystack_documents = response["answer_builder"]["answers"][0].documents
    single_turn['retrieved_contexts'] = [doc.content for doc in haystack_documents]

    evals_list.append(single_turn)

"""
> When constructing the `evals_list`, it is important to align the keys in the single_turn dictionary with the attributes defined in the Ragas [SingleTurnSample](https://docs.ragas.io/en/stable/references/evaluation_schema/#ragas.dataset_schema.SingleTurnSample). This ensures compatibility with the Ragas evaluation framework. Use the retrieved documents and pipeline outputs to populate these fields accurately, as demonstrated in the provided code snippet.

#### Evaluating the pipeline using Ragas EvaluationDataset
The extracted dataset is converted into a Ragas [EvaluationDataset](https://docs.ragas.io/en/stable/references/evaluation_schema/#ragas.dataset_schema.EvaluationDataset) so that Ragas can process it.
We then initialize an LLM evaluator using the HaystackLLMWrapper. Finally, we call Ragas's evaluate() function with our evaluation dataset, three metrics, and the LLM evaluator.
"""
logger.info("#### Evaluating the pipeline using Ragas EvaluationDataset")


evaluation_dataset = EvaluationDataset.from_list(evals_list)

llm = OllamaFunctionCallingAdapterGenerator(model="llama3.2")
evaluator_llm = HaystackLLMWrapper(llm)

result = evaluate(
    dataset=evaluation_dataset,
    metrics=[AnswerRelevancy(), ContextPrecision(), Faithfulness()],
    llm=evaluator_llm,
)

logger.debug(result)
result.to_pandas()

"""
**Haystack Useful Sources**

* [Docs](https://docs.haystack.deepset.ai/docs/intro)
* [Tutorials](https://haystack.deepset.ai/tutorials)
* [Other Cookbooks](https://github.com/deepset-ai/haystack-cookbook)
"""

logger.info("\n\n[DONE]", bright=True)