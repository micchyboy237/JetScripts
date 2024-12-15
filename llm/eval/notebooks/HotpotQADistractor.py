%pip install llama-index-llms-openai!pip install llama-indexfrom llama_index.core.evaluation.benchmarks import HotpotQAEvaluator
from llama_index.core import VectorStoreIndex
from llama_index.core import Document
from llama_index.llms.openai import OpenAI
from llama_index.core.embeddings import resolve_embed_model

llm = OpenAI(model="gpt-3.5-turbo")
embed_model = resolve_embed_model(
    "local:sentence-transformers/all-MiniLM-L6-v2"
)

index = VectorStoreIndex.from_documents(
    [Document.example()], embed_model=embed_model, show_progress=True
)
engine = index.as_query_engine(llm=llm)

HotpotQAEvaluator().run(engine, queries=5, show_result=True)
from llama_index.core.postprocessor import SentenceTransformerRerank

rerank = SentenceTransformerRerank(top_n=3)

engine = index.as_query_engine(
    llm=llm,
    node_postprocessors=[rerank],
)

HotpotQAEvaluator().run(engine, queries=5, show_result=True)
