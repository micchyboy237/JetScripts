from fastapi import FastAPI, HTTPException
from fastapi.routing import APIRouter
from pydantic import BaseModel
from typing import List, Dict
from deeplake.core.vectorstore import VectorStore
from jet.llm.ollama.embeddings import get_ollama_embedding_function
from jet.llm.utils.llama_index_utils import display_jet_source_nodes
from jet.logger import logger
from jet.transformers.formatters import format_json
from llama_index.core.schema import NodeWithScore, TextNode

# FastAPI router
router = APIRouter()

# Configuration
VECTOR_STORE_PATH = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/llm/semantic_search/generated/deeplake/store_1"
EMBEDDING_FUNCTION = get_ollama_embedding_function("mxbai-embed-large")

# Initialize VectorStore
vector_store = VectorStore(
    path=VECTOR_STORE_PATH,
    read_only=True
)

# Request model


class SearchRequest(BaseModel):
    query: str
    top_k: int = 20

# Response model


class SearchResult(BaseModel):
    text: str
    metadata: Dict[str, str]
    score: float


@router.post("/search", response_model=List[SearchResult])
def search_vector_store(request: SearchRequest):
    try:
        logger.info("Received search query")
        logger.debug(format_json(request))

        # Perform search
        results = vector_store.search(
            embedding_data=request.query,
            k=request.top_k,
            embedding_function=EMBEDDING_FUNCTION
        )

        # Parse results into nodes with scores
        nodes_with_scores = [
            NodeWithScore(
                node=TextNode(
                    text=str(text),
                    # Ensure all metadata values are strings
                    metadata={k: str(v) for k, v in metadata.items()}
                ),
                score=float(score)
            )
            for text, metadata, score in zip(results["text"], results["metadata"], results["score"])
        ]

        # Optionally display source nodes (debugging or logging purposes)
        display_jet_source_nodes(request.query, nodes_with_scores)

        # Format response
        response = [
            SearchResult(
                text=node.node.text,
                metadata=node.node.metadata,
                score=node.score
            )
            for node in nodes_with_scores
        ]

        return response

    except Exception as e:
        # Updated logging to avoid 'exc_info'
        logger.error(f"Error during search: {e}")
        raise HTTPException(
            status_code=500, detail="An error occurred during the search.")
