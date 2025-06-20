from sentence_transformers import SentenceTransformer

tokenizer = SentenceTransformer(
    "sentence-transformers/static-retrieval-mrl-en-v1", device="cpu", backend="onnx")
