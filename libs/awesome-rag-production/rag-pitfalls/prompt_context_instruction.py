"""
RAG Pitfalls Demo: Prompt Engineering
Anti-pattern: No Explicit Instruction to Use Context
Solution:     Grounded system prompt with strict context-only instruction

Contrasts hallucination-prone naive prompting with explicit grounding
that forces the LLM to cite retrieved context or refuse.

Reference: rag-pitfalls.md → Prompt Engineering → No Explicit Instruction
"""

from jet.adapters.llama_index.factory import (
    get_llama_cpp_embed_model,
    get_llama_cpp_llm,
)
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.prompts import PromptTemplate

Settings.llm = get_llama_cpp_llm()
Settings.embed_model = get_llama_cpp_embed_model()

documents = [
    Document(
        text="E-4092 indicates a database connection timeout. Fix: increase pool_size to 20 and set pool_recycle=1800."
    ),
    Document(
        text="E-5001 indicates invalid API key format. Keys must be prefixed with 'jk_live_' or 'jk_test_'."
    ),
]

vector_index = VectorStoreIndex.from_documents(documents)
query = "What causes error E-4092 and how do I fix it?"

# ── ❌ ANTI-PATTERN: No Context Instruction ─────────────────────────────────
naive_template = PromptTemplate("Answer: {query}\n\nContext:\n{context_str}")
naive_engine = vector_index.as_query_engine(
    similarity_top_k=3,
    text_qa_template=naive_template,
)

print("=" * 60)
print("❌ NAIVE PROMPT (no grounding instruction)")
print("=" * 60)
naive_response = naive_engine.query(query)
print(f"  Response: {naive_response.response[:300]}")

# ── ✅ SOLUTION: Explicit Grounding Prompt ──────────────────────────────────
grounded_template = PromptTemplate(
    """You are a technical support assistant. Use ONLY the information from the context below to answer the question.
If the context doesn't contain the answer, say "I don't have enough information in my knowledge base."
Always cite the source document when possible.

Context:
{context_str}

Question: {query}
Answer:"""
)
grounded_engine = vector_index.as_query_engine(
    similarity_top_k=3,
    text_qa_template=grounded_template,
)

print("\n" + "=" * 60)
print("✅ GROUNDED PROMPT (strict context-only instruction)")
print("=" * 60)
grounded_response = grounded_engine.query(query)
print(f"  Response: {grounded_response.response[:300]}")
