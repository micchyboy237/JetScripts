import os
import shutil

# create grader for doc retriever
from jet.file import save_file
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from jet.adapters.langchain.chat_ollama import ChatOllama

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(
        os.path.basename(__file__))[0]
)
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


# define a data class
class GradeDocuments(BaseModel):
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


# LLM with function call
llm = ChatOllama(model="llama3.2", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Prompt for the grader
system = """You are a grader assessing relevance of a retrieved document to a user question. 
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. 
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human",
         "Retrieved document: \n\n{document}\n\nUser question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader

# -------- Sample run with streaming --------
sample_doc = """LangChain is a framework for developing applications powered by language models.
It provides components for document retrieval, chain management, and integrations with many model providers."""
sample_question = "What is LangChain used for?"

streamed_chunks = []
for chunk in retrieval_grader.stream({"document": sample_doc, "question": sample_question}):
    streamed_chunks.append(chunk)

# the final structured object is the last chunk
result = streamed_chunks[-1]
save_file(result, f"{OUTPUT_DIR}/result.json")
