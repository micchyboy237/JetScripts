from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain import hub
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.runnables import RunnableLambda
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
# Step-Back Prompting (Question-Answering)

One prompting technique called "Step-Back" prompting can improve performance on complex questions by first asking a "step back" question. This can be combined with regular question-answering applications by then doing retrieval on both the original and step-back question.

Read the paper [here](https://arxiv.org/abs/2310.06117)

See an excellent blog post on this by Cobus Greyling [here](https://cobusgreyling.medium.com/a-new-prompt-engineering-technique-has-been-introduced-called-step-back-prompting-b00e8954cacb)

In this cookbook we will replicate this technique. We modify the prompts used slightly to work better with chat models.
"""
logger.info("# Step-Back Prompting (Question-Answering)")


examples = [
    {
        "input": "Could the members of The Police perform lawful arrests?",
        "output": "what can the members of The Police do?",
    },
    {
        "input": "Jan Sindel’s was born in what country?",
        "output": "what is Jan Sindel’s personal history?",
    },
]
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert at world knowledge. Your task is to step back and paraphrase a question to a more generic step-back question, which is easier to answer. Here are a few examples:""",
        ),
        few_shot_prompt,
        ("user", "{question}"),
    ]
)

question_gen = prompt | ChatOllama(model="llama3.2") | StrOutputParser()

question = "was chatgpt around while trump was president?"

question_gen.invoke({"question": question})


search = DuckDuckGoSearchAPIWrapper(max_results=4)


def retriever(query):
    return search.run(query)

retriever(question)

retriever(question_gen.invoke({"question": question}))




response_prompt = hub.pull("langchain-ai/stepback-answer")

chain = (
    {
        "normal_context": RunnableLambda(lambda x: x["question"]) | retriever,
        "step_back_context": question_gen | retriever,
        "question": lambda x: x["question"],
    }
    | response_prompt
    | ChatOllama(model="llama3.2")
    | StrOutputParser()
)

chain.invoke({"question": question})

"""
## Baseline
"""
logger.info("## Baseline")

response_prompt_template = """You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.

{normal_context}

Original Question: {question}
Answer:"""
response_prompt = ChatPromptTemplate.from_template(response_prompt_template)

chain = (
    {
        "normal_context": RunnableLambda(lambda x: x["question"]) | retriever,
        "question": lambda x: x["question"],
    }
    | response_prompt
    | ChatOllama(model="llama3.2")
    | StrOutputParser()
)

chain.invoke({"question": question})

logger.info("\n\n[DONE]", bright=True)