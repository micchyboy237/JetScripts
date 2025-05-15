import unittest
from chunker import Chunker

class TestChunker(unittest.TestCase):

    def test_init(self):
        # Test the initialization of the Chunker class
        chunker = Chunker(["chunk1", "chunk2", "chunk3"], "query1", "selective")
        self.assertEqual(chunker.chunks, ["chunk1", "chunk2", "chunk3"])
        self.assertEqual(chunker.query, "query1")
        self.assertEqual(chunker.compression_type, "selective")
        self.assertEqual(chunker.model, "meta-llama/Llama-3.2-3B-Instruct")

    def test_compress_chunk(self):
        # Test the compression of a retrieved chunk
        chunker = Chunker(["chunk1", "chunk2", "chunk3"], "query1", "selective")
        chunk = "This is a sample text."
        compressed_chunk = chunker.compress_chunk(chunk)
        self.assertEqual(compressed_chunk, "This is a sample text. You are an expert at information filtering. Your task is to analyze a document chunk and extract ONLY the sentences or paragraphs that are directly relevant to the user's query. Remove all irrelevant content.")

    def test_chunk_text(self):
        # Test the chunking of text
        chunker = Chunker(["chunk1", "chunk2", "chunk3"], "query1", "selective")
        text = "This is a sample text."
        chunks = chunker.chunk_text(text, n=1000, overlap=200)
        self.assertEqual(chunks, ["chunk1", "chunk2", "chunk3"])

    def test_batch_compress_chunks(self):
        # Test the batch compression of chunks
        chunker = Chunker(["chunk1", "chunk2", "chunk3"], "query1", "selective")
        chunks = ["chunk1", "chunk2", "chunk3"]
        query = "query1"
        compression_type = "selective"
        model = "meta-llama/Llama-3.2-3B-Instruct"
        compressed_chunks = chunker.batch_compress_chunks(chunks, query, compression_type, model)
        self.assertEqual(compressed_chunks, ["chunk1", "chunk2", "chunk3"])

if __name__ == "__main__":
    unittest.main()