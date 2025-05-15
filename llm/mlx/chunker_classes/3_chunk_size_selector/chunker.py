from typing import List, Union, Dict
import fitz


class Chunker:
    """
    A reusable class for chunking text into overlapping segments for RAG pipelines.

    Attributes:
        chunk_size (int): Number of characters per chunk.
        overlap (int): Number of overlapping characters between chunks.
        strategy (str): Chunking strategy ('character' or extendable to 'sentence', etc.).
    """

    def __init__(self, chunk_size: int = 256, overlap: int = None, strategy: str = "character"):
        """
        Initialize the Chunker with specified parameters.

        Args:
            chunk_size (int): Number of characters per chunk (default: 256).
            overlap (int, optional): Overlapping characters between chunks. Defaults to chunk_size // 5.
            strategy (str): Chunking strategy (default: 'character').
        """
        self.chunk_size = chunk_size
        self.overlap = overlap if overlap is not None else chunk_size // 5
        self.strategy = strategy

        if self.chunk_size <= 0:
            raise ValueError("Chunk size must be positive.")
        if self.overlap < 0:
            raise ValueError("Overlap must be non-negative.")
        if self.overlap >= self.chunk_size:
            raise ValueError("Overlap must be less than chunk size.")
        if self.strategy not in ["character"]:
            raise ValueError(f"Unsupported chunking strategy: {self.strategy}")

    def chunk_text(self, text: str) -> List[str]:
        """
        Splits text into overlapping chunks based on the specified strategy.

        Args:
            text (str): The text to be chunked.

        Returns:
            List[str]: A list of text chunks.
        """
        if not text:
            return []

        if self.strategy == "character":
            return self._chunk_by_characters(text)

        # Placeholder for future strategies (e.g., sentence-based)
        raise NotImplementedError(
            f"Chunking strategy '{self.strategy}' not implemented.")

    def _chunk_by_characters(self, text: str) -> List[str]:
        """
        Splits text into overlapping chunks based on character counts.

        Args:
            text (str): The text to be chunked.

        Returns:
            List[str]: A list of text chunks.
        """
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.overlap):
            chunks.append(text[i:i + self.chunk_size])
        return chunks

    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """
        Extracts text from a PDF file.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            str: Extracted text from the PDF.
        """
        try:
            mypdf = fitz.open(pdf_path)
            all_text = ""
            for page in mypdf:
                all_text += page.get_text("text") + " "
            return all_text.strip()
        except Exception as e:
            raise RuntimeError(f"Failed to extract text from PDF: {str(e)}")

    def chunk_multiple_sizes(self, text: str, chunk_sizes: List[int], overlaps: Union[int, List[int], None] = None) -> Dict[int, List[str]]:
        """
        Chunks text using multiple chunk sizes and overlaps.

        Args:
            text (str): The text to be chunked.
            chunk_sizes (List[int]): List of chunk sizes to evaluate.
            overlaps (Union[int, List[int], None]): Single overlap value or list of overlaps corresponding to chunk sizes.

        Returns:
            Dict[int, List[str]]: Dictionary mapping chunk sizes to their respective chunks.
        """
        if not chunk_sizes:
            raise ValueError("Chunk sizes list cannot be empty.")

        # Handle overlap input
        if overlaps is None:
            overlaps = [size // 5 for size in chunk_sizes]
        elif isinstance(overlaps, int):
            overlaps = [overlaps] * len(chunk_sizes)
        elif len(overlaps) != len(chunk_sizes):
            raise ValueError(
                "Length of overlaps must match length of chunk sizes.")

        # Create chunks for each chunk size
        text_chunks_dict = {}
        for size, overlap in zip(chunk_sizes, overlaps):
            chunker = Chunker(chunk_size=size, overlap=overlap,
                              strategy=self.strategy)
            text_chunks_dict[size] = chunker.chunk_text(text)

        return text_chunks_dict
