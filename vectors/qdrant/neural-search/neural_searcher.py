# Import necessary libraries
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from qdrant_client.models import Filter
from jet.llm import get_model_path


class NeuralSearcher:
    def __init__(self, collection_name):
        # Store the collection name
        self.collection_name = collection_name
        # Initialize encoder model

        model_path = get_model_path("sentence-transformers/all-MiniLM-L6-v2")
        self.model = SentenceTransformer(model_path)
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient("http://localhost:6333")

    # Define the search function
    def search(self, text: str):
        # Convert the text query into a vector
        vector = self.model.encode(text).tolist()

        # Perform the search using Qdrant client
        search_result = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=None,  # No filter applied for now
            limit=5,  # Limit to 5 closest results
        ).points

        # Extract payload from search results
        payloads = [hit.payload for hit in search_result]
        return payloads

    # Define function with search filters
    def search_with_filter(self, text: str, city_of_interest: str):
        # Convert the text query into a vector
        vector = self.model.encode(text).tolist()

        # Define a filter for cities
        city_filter = Filter(**{
            "must": [{
                "key": "city",  # Field for city information
                "match": {  # Condition to match the city value
                    "value": city_of_interest
                }
            }]
        })

        # Perform the search with the city filter
        search_result = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=city_filter,  # Apply the city filter
            limit=5  # Limit to 5 closest results
        ).points

        # Extract payload from search results
        payloads = [hit.payload for hit in search_result]
        return payloads

# Now the class NeuralSearcher is ready for use.
