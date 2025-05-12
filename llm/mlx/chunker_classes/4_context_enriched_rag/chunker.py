```python
class Chunker:
    """
    A class for chunking text into segments of n characters with overlap.
    """

    def __init__(self, text, n, overlap):
        """
        Initializes the Chunker class.

        Args:
        text (str): The text to be chunked.
        n (int): The number of characters in each chunk.
        overlap (int): The number of overlapping characters between chunks.
        """
        self.text = text
        self.n = n
        self.overlap = overlap

    def chunk(self):
        """
        Chunks the text into segments of n characters with overlap.

        Returns:
        list: A list of text chunks.
        """
        chunks = []
        for i in range(0, len(self.text), self.n - self.overlap):
            chunks.append(self.text[i:i + self.n])
        return chunks

    def create_embeddings(self, model="BAAI/bge-en-icl"):
        """
        Creates embeddings for the given text using the specified OpenAI model.

        Args:
        model (str): The model to be used for creating embeddings. Default is "BAAI/bge-en-icl".

        Returns:
        dict: The response from the OpenAI API containing the embeddings.
        """
        response = client.embeddings.create(
            model=model,
            input=self.text
        )
        return response

def cosine_similarity(vec1, vec2):
    """
    Calculates the cosine similarity between two vectors.

    Args:
    vec1 (np.ndarray): The first vector.
    vec2 (np.ndarray): The second vector.

    Returns:
    float: The cosine similarity between the two vectors.
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
```