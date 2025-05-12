```python
def chunk_text(text, n, overlap):
    """
    Chunks the given text into segments of n characters with overlap.
    
    Args:
        text (str): The text to be chunked.
        n (int): The number of characters in each chunk.
        overlap (int): The number of overlapping characters between chunks.
    
    Returns:
        List[str]: A list of text chunks.
    """
    chunks = []  # Initialize an empty list to store the chunks
    # Loop through the text with a step size of (n - overlap)
    for i in range(0, len(text), n - overlap):
        # Append a chunk of text from index i to i + n to the chunks list
        chunks.append(text[i:i + n])
    return chunks  # Return the list of text chunks

def process_document(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    Process a document for use with adaptive retrieval.
    
    Args:
        pdf_path (str): Path to the PDF file.
        chunk_size (int): Size of each chunk in characters.
        chunk_overlap (int): Overlap between chunks in characters.
    
    Returns:
        Tuple[List[str], SimpleVectorStore]: Document chunks and vector store.
    """
    # Extract text from the PDF file
    extracted_text = extract_text_from_pdf(pdf_path)
    # Chunk the extracted text
    chunks = chunk_text(extracted_text, chunk_size, chunk_overlap)
    # Create embeddings for the text chunks
    chunk_embeddings = create_embeddings(chunks)
    # Initialize the vector store
    store = SimpleVectorStore()
    # Add each chunk and its embedding to the vector store with metadata
    for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
        store.add_item(
            text=chunk,
            embedding=embedding,
            metadata={"index": i, "source": pdf_path}
        )
    # Return the chunks and the vector store
    return chunks, store

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file and prints the first `num_chars` characters.
    
    Args:
        pdf_path (str): Path to the PDF file.
    
    Returns:
        str: Extracted text from the PDF.
    """
    # Open the PDF file
    mypdf = fitz.open(pdf_path)
    all_text = ""  # Initialize an empty string to store the extracted text
    # Iterate through each page in the PDF
    for page_num in range(mypdf.page_count):
        page = mypdf[page_num]  # Get the page
        text = page.get_text("text")  # Extract text from the page
        all_text += text  # Append the extracted text to the all_text string
    return all_text  # Return the extracted text

def create_embeddings(text, model="BAAI/bge-en-icl"):
    """
    Creates embeddings for the given text.
    
    Args:
        text (str or List[str]): The input text(s) for which embeddings are to be created.
        model (str): The model to be used for creating embeddings.
    
    Returns:
        List[float] or List[List[float]]: The embedding vector(s).
    """
    # Handle both string and list inputs by converting string input to a list
    input_text = text if isinstance(text, list) else [text]
    # Create embeddings for the input text using the specified model
    response = client.embeddings.create(
        model=model,
        input=input_text
    )
    # If the input was a single string, return just the first embedding
    if isinstance(text, str):
        return response.data[0].embedding
    # Otherwise, return all embeddings for the list of texts
    return [item.embedding for item in response.data]

class SimpleVectorStore:
    """
    A simple vector store implementation using NumPy.
    """
    def __init__(self):
        """
        Initialize the vector store.
        """
        self.vectors = []  # List to store embedding vectors
        self.texts = []  # List to store original texts
        self.metadata = []  # List to store metadata for each text

    def add_item(self, text, embedding, metadata=None):
        """
        Add an item to the vector store.
        
        Args:
            text (str): The original text.
            embedding (List[float]): The embedding vector.
            metadata (dict, optional): Additional metadata.
        """
        self.vectors.append(np.array(embedding))  # Convert embedding to numpy array and add to vectors list
        self.texts.append(text)  # Add the original text to texts list
        self.metadata.append(metadata or {})  # Add metadata to metadata list, default to empty dict if None

    def similarity_search(self, query_embedding, k=5, filter_func=None):
        """
        Find the most similar items to a query embedding.
        
        Args:
            query_embedding (List[float]): Query embedding vector.
            k (int): Number of results to return.
            filter_func (callable, optional): Function to filter results.
        
        Returns:
            List[Dict]: Top k most similar items with their texts and metadata.
        """
        if not self.vectors:
            return