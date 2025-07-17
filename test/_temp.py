from sentence_transformers import SentenceTransformer

from jet.models.model_registry.transformers.sentence_transformer_registry import SentenceTransformerRegistry

# model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5",
#                             device="cpu", backend="onnx", trust_remote_code=True)
model = SentenceTransformerRegistry.load_model(
    "nomic-ai/nomic-embed-text-v1.5")
sentences = [
    'search_document: TSNE is a dimensionality reduction algorithm created by Laurens van Der Maaten']
embeddings = model.encode(sentences)
print(embeddings)
