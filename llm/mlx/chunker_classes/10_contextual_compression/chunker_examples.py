from chunker import Chunker

class ChunkerExample:
    def __init__(self):
        self.chunker = Chunker(
            chunks=["This is a sample text.", "This is another sample text."],
            query="sample text",
            compression_type="summary",
            model="meta-llama/Llama-3.2-3B-Instruct"
        )

    def compress_chunk(self, chunk):
        return self.chunker.compress_chunk(chunk)

    def chunk_text(self, text, n=1000, overlap=200):
        return self.chunker.chunk_text(text, n, overlap)

    def batch_compress_chunks(self, chunks, query, compression_type="selective", model="meta-llama/Llama-3.2-3B-Instruct"):
        return self.chunker.batch_compress_chunks(chunks, query, compression_type, model)

    def test_example(self):
        result = self.chunker.compress_chunk("This is a sample text.")
        print(result)

        chunks = self.chunker.chunk_text("This is a sample text.", n=1000, overlap=200)
        print(chunks)

        result = self.chunker.batch_compress_chunks(chunks, query="sample text", compression_type="extraction", model="meta-llama/Llama-3.2-3B-Instruct")
        print(result)

    def main(self):
        result = self.test_example()
        print(result)

        chunks = self.chunker.chunk_text("This is a sample text.", n=1000, overlap=200)
        print(chunks)

        result = self.chunker.batch_compress_chunks(chunks, query="sample text", compression_type="extraction", model="meta-llama/Llama-3.2-3B-Instruct")
        print(result)

if __name__ == "__main__":
    chunker_example = ChunkerExample()
    chunker_example.main()