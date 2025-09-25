from deepeval.metrics import ContextualRelevancyMetric
from deepeval.tracing import evaluate_thread
from deepeval.tracing import observe, update_current_span, update_current_trace
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
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
---
id: evals-in-prod
title: Setup Evals in Prod
sidebar_label: Setup Evals in Prod
---

In this section we'll learn how to set up tracing for our medical chatbot to observe it on a component level and ensure your chatbot performs well and gets full visibilty for debugging internal components.

In the development section of this tutorial, we've already added `@observe` decorator to our chatbot's components, now we will add metrics and spans to this tracing setup to enable evaluations.

## Setup Tracing

`deepeval` offers an `@observe` decorator for you to apply metrics at any point in your LLM app to evaluate any [LLM interaction](https://deepeval.com/docs/evaluation-test-cases#what-is-an-llm-interaction),
this provides full visibility for debugging internal components of your LLM application. [Learn more about tracing here](https://deepeval.com/docs/evaluation-llm-tracing).

To add metrics and spans to your traces, modify your `MedicalChatbot` class like this:
"""
logger.info("## Setup Tracing")


class MedicalChatbot:
    def __init__(
        self,
        document_path,
        model="llama3.2",
        encoder="all-MiniLM-L6-v2",
        memory=":memory:",
        system_prompt=""
    ):
        self.model = ChatOllama(model="llama3.2")
        self.appointments = {}
        self.encoder = SentenceTransformer(encoder)
        self.client = QdrantClient(memory)
        self.store_data(document_path)
        self.system_prompt = system_prompt or (
            "You are a virtual health assistant designed to support users with symptom understanding and appointment management. Start every conversation by actively listening to the user's concerns. Ask clear follow-up questions to gather information like symptom duration, intensity, and relevant health history. Use available tools to fetch diagnostic information or manage medical appointments. Never assume a diagnosis unless there's enough detail, and always recommend professional medical consultation when appropriate."
        )
        self.setup_agent(self.system_prompt)

    def store_data(self, document_path):
        ...

    @tool
    @observe(metrics=[ContextualRelevancyMetric()], type="retriever")
    def query_engine(self, query: str) -> str:
        """"A tool to retrive data on various diagnosis methods from gale encyclopedia"""
        hits = self.client.search(
            collection_name="gale_encyclopedia",
            query_vector=self.encoder.encode(query).tolist(),
            limit=3,
        )

        contexts = [hit.payload['content'] for hit in hits]

        update_current_span(
            input=query,
            retrieval_context=contexts
        )
        return "\n".join(contexts)

    ... # Other tools here

    @observe(type="agent")
    def interactive_session(self, session_id):
        logger.debug("Hello! I am Baymax, your personal health care companian.")
        logger.debug("Please enter your symptoms or ask about appointment details. Type 'exit' to quit.")

        while True:
            user_input = input("Your query: ")
            if user_input.lower() == 'exit':
                break

            response = self.agent_with_chat_history.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            update_current_trace(
                thread_id=session_id,
                input=user_input,
                output=response["output"]
            )
            logger.debug("Agent Response:", response["output"])

"""
This tracing setup is done for the `interactive_session()` method, for your chatbot in production, you would observe your main callback function. Here's the docs to [learn more about tracing](https://deepeval.com/docs/evaluation-llm-tracing).

:::tip
Adding `@observe` tag to all your functions is also helpul in evaluating your entire workflow, this also does not interrupt your application. You can see the entire workflow with just a single line of code.
:::

## Evaluating Spans

From the previous tracing code we've seen how to setup trace spans, here's how you can evaluate those spans:
"""
logger.info("## Evaluating Spans")

...
...

@observe(type="agent")
def interactive_session(self, session_id):
    logger.debug("Hello! I am Baymax, your personal health care companian.")
    logger.debug("Please enter your symptoms or ask about appointment details. Type 'exit' to quit.")

    while True:
        user_input = input("Your query: ")
        if user_input.lower() == 'exit':
            break

        response = self.agent_with_chat_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        update_current_trace(
            thread_id=session_id, # Keep your unique <thread id> here
            input=user_input,
            output=response["output"]
        )
        logger.debug("Agent Response:", response["output"])

"""
You can now use this thread id to evaluate this trace with the following code:
"""
logger.info("You can now use this thread id to evaluate this trace with the following code:")


evaluate_thread(thread_id="your-thread-id", metric_collection="Metric Collection")

"""
You can create a metric collection on the Confident AI platform to run online evaluations and catch regression or bugs, [learn more here](https://www.confident-ai.com/docs/metrics/metric-collections).

And that's it! You now have a reliable medical chatbot with component level tracing with just a few lines of code.

:::tip Next Steps
Setup [Confident AI](https://deepeval.com/tutorials/tutorial-setup) to track your medical chatbot's performance across builds, regressions, and evolving datasets. **It's free to get started.** _(No credit card required)_

Learn more [here](https://www.confident-ai.com).
:::
"""
logger.info("You can create a metric collection on the Confident AI platform to run online evaluations and catch regression or bugs, [learn more here](https://www.confident-ai.com/docs/metrics/metric-collections).")

logger.info("\n\n[DONE]", bright=True)