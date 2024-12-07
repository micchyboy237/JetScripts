from txtai import LLM, RAG
from txtai.pipeline import Tokenizer, Similarity
from jet.llm.embeddings import Embeddings
from jet.llm.search import (
    load_local_json,
)


def main():
    embedding_model = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    # Example setup (you would need to initialize the similarity model, tokenizer, etc.)
    # Assuming you have a similarity model
    similarity_instance = Similarity(path=embedding_model)
    # path_to_model = "path/to/your/model"
    tokenizer = Tokenizer()  # Or use default if not specified

    # embeddings = Embeddings()
    # embeddings.load(provider="huggingface-hub",
    #                 container="neuml/txtai-wikipedia")
    llm = LLM(path="ollama/llama3.1", method="litellm",
              api_base="http://0.0.0.0:4000")

    # Initialize the RAG pipeline
    rag_pipeline = RAG(
        # embeddings,
        similarity=similarity_instance,
        path=llm,
        tokenizer=tokenizer,
        context=3,  # Number of context matches to consider
        output="default"  # Can be 'default', 'flatten', or 'reference'
    )

    # Sample input data
    question_queue = [
        # {"name": "query1", "query": "How to set this up and run? Provide sample usage.",
        #     "question": "How to set this up and run? Provide sample usage.",
        #     "snippet": "Paris is a beautiful city."},
        # {"name": "query2", "query": "How to create crewAI agents with tasks?",
        #     "question": "How to create crewAI agents with tasks?",
        #     "snippet": "Jupiter is the largest planet."}
        "How to set this up and run? Provide sample usage.",
        "How to create crewAI agents with tasks?",
    ]

    # Prepare RAG texts
    dataset_path = "/Users/jethroestrada/Desktop/External_Projects/AI/agents_2/crewAI/my_project/src/my_project/generated/rag/crewai-docs.json"
    dataset = load_local_json(dataset_path)
    texts = [row["page_content"] for row in dataset]

    # Call the RAG pipeline to get answers for the questions
    results = rag_pipeline(queue=question_queue,
                           texts=texts, stream=True, maxlength=2048)

    # Print the results
    for result in results:
        print(f"Name: {result['name']}")
        print(f"Answer: {result['answer']}")
        print(f"Reference: {result.get('reference', 'No reference')}")
        print("-" * 50)


if __name__ == "__main__":
    main()
