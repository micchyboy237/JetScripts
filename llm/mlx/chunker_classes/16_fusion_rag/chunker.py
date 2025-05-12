```python
class Chunker:
    def __init__(self, pdf_path, chunk_size=1000, chunk_overlap=200):
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store = None
        self.bm25_index = None

    def process_document(self):
        text = extract_text_from_pdf(self.pdf_path)
        chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
        self.vector_store = SimpleVectorStore()
        self.vector_store.add_items(chunks, create_embeddings([chunk["text"] for chunk in chunks]))
        self.bm25_index = create_bm25_index(chunks)
        return self.vector_store, self.bm25_index

    def answer_with_fusion_rag(self, query, chunks, vector_store, bm25_index, k=5, alpha=0.5):
        retrieved_docs = fusion_retrieval(query, chunks, vector_store, bm25_index, k, alpha)
        context = "nn---nn".join([doc["text"] for doc in retrieved_docs])
        response = generate_response(query, context)
        return {
            "query": query,
            "retrieved_documents": retrieved_docs,
            "response": response
        }

    def create_embeddings(self, texts, model="BAAI/bge-en-icl"):
        input_texts = texts if isinstance(texts, list) else [texts]
        batch_size = 100
        all_embeddings = []
        for i in range(0, len(input_texts), batch_size):
            batch = input_texts[i:i + batch_size]
            response = client.embeddings.create(model=model, input=batch)
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        if isinstance(texts, str):
            return all_embeddings[0]
        else:
            return all_embeddings

    def extract_text_from_pdf(self, pdf_path):
        # Implement your own extract_text_from_pdf function
        pass

    def clean_text(self, text):
        # Implement your own clean_text function
        pass

    def chunk_text(self, text, chunk_size, chunk_overlap):
        # Implement your own chunk_text function
        pass

    def create_bm25_index(self, chunks):
        # Implement your own create_bm25_index function
        pass

    def create_vector_store(self):
        # Implement your own create_vector_store function
        pass

    def evaluate_fusion_retrieval(self, pdf_path, test_queries, reference_answers=None, k=5, alpha=0.5):
        # Implement your own evaluate_fusion_retrieval function
        pass

    def compare_retrieval_methods(self, query, chunks, vector_store, bm25_index, k=5, alpha=0.5, reference_answer=None):
        # Implement your own compare_retrieval_methods function
        pass

    def fusion_retrieval(self, query, chunks, vector_store, bm25_index, k=5, alpha=0.5):
        # Implement your own fusion_retrieval function
        pass
```