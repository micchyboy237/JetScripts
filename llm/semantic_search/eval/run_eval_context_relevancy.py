from jet.file.utils import load_file, save_file
from jet.llm.ollama.base import Ollama


if __name__ == "__main__":
    llm_model = "gemma3:4b"
    query = "What are the steps in registering a National ID in the Philippines?"

    data_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/generated/run_llm_reranker/final_llm_reranker_results.json"
    output_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/semantic_search/eval/generated/run_eval_context_relevancy"
    data = load_file(data_file)

    llm = Ollama(model=llm_model, temperature=0.3)

    context = "\n\n".join([d["text"] for d in data])
    response = llm.chat(query, context=context)
    llm_chat_history_file = f"{output_dir}/llm_chat_history.md"
    chat_history_list = [
        f"## Query\n\n{query}",
        f"## Context\n\n{context}",
        f"## Response\n\n{response}",
    ]
    chat_history = "\n\n".join(chat_history_list)
    save_file(chat_history, llm_chat_history_file)
