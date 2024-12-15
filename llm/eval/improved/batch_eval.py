%pip install llama-index-llms-openai llama-index-embeddings-openaiimport nest_asyncio

nest_asyncio.apply()
import os
import openai

os.environ["OPENAI_API_KEY"] = "sk-..."
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Response
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    CorrectnessEvaluator,
)
from llama_index.core.node_parser import SentenceSplitter
import pandas as pd

pd.set_option("display.max_colwidth", 0)
gpt4 = OpenAI(temperature=0, model="gpt-4")

faithfulness_gpt4 = FaithfulnessEvaluator(llm=gpt4)
relevancy_gpt4 = RelevancyEvaluator(llm=gpt4)
correctness_gpt4 = CorrectnessEvaluator(llm=gpt4)
documents = SimpleDirectoryReader("./test_wiki_data/").load_data()
llm = OpenAI(temperature=0.3, model="gpt-3.5-turbo")
splitter = SentenceSplitter(chunk_size=512)
vector_index = VectorStoreIndex.from_documents(
    documents, transformations=[splitter]
)
%pip install spacy datasets span-marker scikit-learnfrom llama_index.core.evaluation import DatasetGenerator

dataset_generator = DatasetGenerator.from_documents(documents, llm=llm)

qas = dataset_generator.generate_dataset_from_nodes(num=3)
from llama_index.core.evaluation import BatchEvalRunner

runner = BatchEvalRunner(
    {"faithfulness": faithfulness_gpt4, "relevancy": relevancy_gpt4},
    workers=8,
)

eval_results = await runner.aevaluate_queries(
    vector_index.as_query_engine(llm=llm), queries=qas.questions
)



print(len([qr for qr in qas.qr_pairs]))
print(eval_results.keys())

print(eval_results["faithfulness"][0].dict().keys())

print(eval_results["faithfulness"][0].passing)
print(eval_results["faithfulness"][0].response)
print(eval_results["faithfulness"][0].contexts)
def get_eval_results(key, eval_results):
    results = eval_results[key]
    correct = 0
    for result in results:
        if result.passing:
            correct += 1
    score = correct / len(results)
    print(f"{key} Score: {score}")
    return score
score = get_eval_results("faithfulness", eval_results)
score = get_eval_results("relevancy", eval_results)
