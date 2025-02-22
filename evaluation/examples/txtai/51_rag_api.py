from fastapi import FastAPI, APIRouter
from jet.libs.txtai.api import application, Extension
import requests
from datasets import load_dataset

# Define RAG API extension


class RAG(Extension):
    def __call__(self, app):
        app.include_router(RAGRouter().router)

# API router for the RAG endpoint


class RAGRouter:
    router = APIRouter()

    @staticmethod
    @router.get("/rag")
    def rag(text: str):
        """
        Runs a retrieval augmented generation (RAG) pipeline.

        Args:
            text: input text

        Returns:
            response
        """
        # Run embeddings search
        results = application.get().search(text, 3)
        context = " ".join([x["text"] for x in results])

        prompt = f"""
        Answer the following question using only the context below.

        Question: {text}
        Context: {context}
        """

        return {"response": application.get().pipeline("llm", (prompt,))}

# Initialize FastAPI app and add the RAG extension


def create_app():
    app = FastAPI()
    RAG()(app)
    return app

# Function to create the embeddings database


def create_embeddings_db():
    # Load dataset
    ds = load_dataset("ag_news", split="train")

    # API endpoint and headers
    url = "http://localhost:8000"
    headers = {"Content-Type": "application/json"}

    # Add data in batches to the API
    batch = []
    for text in ds["text"]:
        batch.append({"text": text})
        if len(batch) == 4096:
            requests.post(f"{url}/add", headers=headers,
                          json=batch, timeout=120)
            batch = []
    if batch:
        requests.post(f"{url}/add", headers=headers, json=batch, timeout=120)

    # Build index
    requests.get(f"{url}/index")

# Function to call the RAG endpoint


def run_rag_query(text):
    url = "http://localhost:8000"
    response = requests.get(f"{url}/rag?text={text}").json()
    return response["response"]

# Main function to start the API and perform queries


def main():
    # Start the FastAPI app (for development purposes)
    import uvicorn
    from threading import Thread

    def run_api():
        uvicorn.run(create_app(), host="0.0.0.0", port=8000)

    api_thread = Thread(target=run_api)
    api_thread.start()

    # Allow the API to start up
    import time
    time.sleep(60)

    # Create the embeddings database
    create_embeddings_db()

    # Run sample queries
    queries = [
        "Who is the current President?",
        "Who lost the presidential election?",
        "Who won the World Series?",
        "Who did the Red Sox beat to win the world series?",
        "What major hurricane hit the USA?",
        "What mobile phone manufacturer has the largest current marketshare?"
    ]

    for query in queries:
        print(f"Query: {query}")
        print("Response:", run_rag_query(query))


if __name__ == "__main__":
    main()
