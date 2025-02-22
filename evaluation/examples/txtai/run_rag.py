from jet.libs.txtai import Embeddings, RAG, LLM
from jet.logger import logger
from jet.llm.search import load_local_json

dataset_path = "/Users/jethroestrada/Desktop/External_Projects/AI/agents_2/crewAI/my_project/src/my_project/generated/rag/crewai-docs.json"
dataset = load_local_json(dataset_path)
data = [row["page_content"] for row in dataset]

# Input data
# data = [
#     "US tops 5 million confirmed virus cases",
#     "Canada's last fully intact ice shelf has suddenly collapsed, " +
#     "forming a Manhattan-sized iceberg",
#     "Beijing mobilises invasion craft along coast as Taiwan tensions escalate",
#     "The National Park Service warns against sacrificing slower friends " +
#     "in a bear attack",
#     "Maine man wins $1M from $25 lottery ticket",
#     "Make huge profits without work, earn up to $100,000 a day"
# ]

# Build embeddings index
embeddings = Embeddings(content=True)
embeddings.index(data)

# Create and run pipeline
llm = LLM(path="ollama/llama3.1", method="litellm",
          api_base="http://localhost:11434")
rag = RAG(embeddings, llm, template="""
  Answer the following question using the provided context.

  Question:
  {question}

  Context:
  {context}
""")

result = rag("What is crew AI?", maxlength=4096)
logger.success(result['answer'])
