class Chunker:
    """
    Chunker class for compressing text chunks.

    Attributes:
    chunks (List[str]): List of text chunks to compress
    query (str): User query
    compression_type (str): Type of compression ("selective", "summary", or "extraction")
    model (str): LLM model to use
    """

    def __init__(self, chunks, query, compression_type="selective", model="meta-llama/Llama-3.2-3B-Instruct"):
        """
        Initialize the Chunker class.

        Args:
        chunks (List[str]): List of text chunks to compress
        query (str): User query
        compression_type (str): Type of compression ("selective", "summary", or "extraction")
        model (str): LLM model to use
        """
        self.chunks = chunks
        self.query = query
        self.compression_type = compression_type
        self.model = model

    def compress_chunk(self, chunk):
        """
        Compress a retrieved chunk by keeping only the parts relevant to the query.

        Args:
        chunk (str): Text chunk to compress
        query (str): User query
        compression_type (str): Type of compression ("selective", "summary", or "extraction")
        model (str): LLM model to use

        Returns:
        str: Compressed chunk
        """
        # Define system prompts for different compression approaches
        if self.compression_type == "selective":
            system_prompt = """You are an expert at information filtering. Your task is to analyze a document chunk and extract ONLY the sentences or paragraphs that are directly
relevant to the user's query. Remove all irrelevant content."""
        elif self.compression_type == "summary":
            system_prompt = """You are an expert at summarization. Your task is to create a concise summary of the provided chunk that focuses ONLY on
information relevant to the user's query."""
        else:  # extraction
            system_prompt = """You are an expert at information extraction. Your task is to extract ONLY the exact sentences from the document chunk that contain information relevant
to answering the user's query."""

        # Define the user prompt with the query and document chunk
        user_prompt = f"""
        Query: {self.query}
        Document Chunk:
        {chunk}
        Extract only the content relevant to answering this query.
        """

        # Generate context from the compressed chunk
        context = "nn---nn".join(self.compress_chunk(chunk))

        # Generate a response based on the compressed chunk
        response = generate_response(self.query, context, self.model)

        # Prepare the result dictionary
        result = {
            "query": self.query,
            "compressed_chunk": self.compress_chunk(chunk),
            "context_length_reduction": f"{sum(self.compress_chunk(chunk)) / len(self.compress_chunk(chunk)):.2f}%",
            "response": response
        }

        return result

    def chunk_text(self, text, n=1000, overlap=200):
        """
        Chunk the given text into segments of n characters with overlap.

        Args:
        text (str): The text to be chunked. n (int): The number of characters in each chunk. overlap (int): The number of overlapping characters between chunks.

        Returns:
        List[str]: A list of text chunks.
        """
        chunks = []
        for i in range(0, len(text), n - overlap):
            chunks.append(text[i:i + n])
        return chunks

    def batch_compress_chunks(self, chunks, query, compression_type="selective", model="meta-llama/Llama-3.2-3B-Instruct"):
        """
        Compress multiple chunks individually.
        
        Args:
        chunks (List[str]): List of text chunks to compress
        query (str): User query
        compression_type (str): Type of compression ("selective", "summary", or "extraction")
