import sys

from jet.llm.ollama.base import Ollama
from retriever.retrieve import hybrid_search
from llama_index.core.base.llms.types import ChatResponse


project_path = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/test/chatgpt/anime_scraper"
sys.path.insert(0, project_path)


def generate_answer(query):
    documents = hybrid_search(query)
    context = "\n\n".join([doc[1] for doc in documents])

    prompt = f"Answer the question based on the following context:\n\n{context}\n\nQ: {query}\nA:"
    llm = Ollama(model="llama3.1", stream=True)
    response: ChatResponse = llm.chat(prompt)

    return response.message.content


if __name__ == "__main__":
    query = input("Enter your query: ")
    answer = generate_answer(query)
    print("\n🔹 AI Response:\n", answer)
