from PIL import Image
from jet.llm.mlx.base import MLX
from jet.llm.utils.embeddings import get_embedding_function
from jet.logger import CustomLogger
import base64
import pypdf
import io
import json
import numpy as np
import os
import re
import shutil
import tempfile

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(
    script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")
DATA_DIR = os.path.join(script_dir, "data")
os.makedirs(DATA_DIR, exist_ok=True)
logger.info("Initializing MLX and embedding function")
mlx = MLX()
embed_func = get_embedding_function("mxbai-embed-large")


def extract_content_from_pdf(pdf_path, output_dir=None):
    temp_dir = None
    if output_dir is None:
        temp_dir = tempfile.mkdtemp()
        output_dir = temp_dir
    else:
        os.makedirs(output_dir, exist_ok=True)
    text_data = []
    image_paths = []
    logger.debug(f"Extracting content from {pdf_path}...")
    try:
        with open(pdf_path, "rb") as file:
            reader = pypdf.PdfReader(file)
            for page_number in range(len(reader.pages)):
                page = reader.pages[page_number]
                text = page.extract_text().strip() or ""
                if text:
                    text_data.append({
                        "content": text,
                        "metadata": {
                            "source": pdf_path,
                            "page": page_number + 1,
                            "type": "text"
                        }
                    })
                images = page.images
                for img_index, img in enumerate(images):
                    img_filename = f"page_{page_number+1}_img_{img_index+1}.png"
                    img_path = os.path.join(output_dir, img_filename)
                    with open(img_path, "wb") as img_file:
                        img_file.write(img.data)
                    image_paths.append({
                        "path": img_path,
                        "metadata": {
                            "source": pdf_path,
                            "page": page_number + 1,
                            "image_index": img_index + 1,
                            "type": "image"
                        }
                    })
        logger.debug(
            f"Extracted {len(text_data)} text segments and {len(image_paths)} images")
        return text_data, image_paths
    except Exception as e:
        logger.debug(f"Error extracting content: {e}")
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise


def chunk_text(text_data, chunk_size=1000, overlap=200):
    chunked_data = []
    for item in text_data:
        text = item["content"]
        metadata = item["metadata"]
        if len(text) < chunk_size / 2:
            chunked_data.append({
                "content": text,
                "metadata": metadata
            })
            continue
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if chunk:
                chunks.append(chunk)
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = i
            chunk_metadata["chunk_count"] = len(chunks)
            chunked_data.append({
                "content": chunk,
                "metadata": chunk_metadata
            })
    logger.debug(f"Created {len(chunked_data)} text chunks")
    return chunked_data


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read())
        return encoded_image.decode('utf-8')


def generate_image_caption(image_path, model="llava-hf/llava-1.5-7b-hf"):
    if not os.path.exists(image_path):
        return "Error: Image file not found"
    try:
        Image.open(image_path)
        base64_image = encode_image(image_path)
        system_prompt = """You are an assistant specialized in describing images from academic papers. 
Provide detailed captions for the image that capture key information. 
If the image contains charts, tables, or diagrams, describe their content and purpose clearly. 
Your caption should be optimized for future retrieval when people ask questions about this content."""
        response = mlx.chat(
            [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in detail, focusing on its academic content:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            model=model,
            max_tokens=300
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error generating caption: {str(e)}"


def process_images(image_paths, model="llava-hf/llava-1.5-7b-hf"):
    image_data = []
    logger.debug(f"Generating captions for {len(image_paths)} images...")
    for i, img_item in enumerate(image_paths):
        logger.debug(f"Processing image {i+1}/{len(image_paths)}...")
        img_path = img_item["path"]
        metadata = img_item["metadata"]
        caption = generate_image_caption(img_path, model)
        image_data.append({
            "content": caption,
            "metadata": metadata,
            "image_path": img_path
        })
    return image_data


class MultiModalVectorStore:
    def __init__(self):
        self.vectors = []
        self.contents = []
        self.metadata = []

    def add_item(self, content, embedding, metadata=None):
        self.vectors.append(np.array(embedding))
        self.contents.append(content)
        self.metadata.append(metadata or {})

    def add_items(self, items, embeddings):
        for item, embedding in zip(items, embeddings):
            self.add_item(
                content=item["content"],
                embedding=embedding,
                metadata=item.get("metadata", {})
            )

    def similarity_search(self, query_embedding, k=5):
        if not self.vectors:
            return []
        query_vector = np.array(query_embedding).flatten()
        similarities = []
        for i, vector in enumerate(self.vectors):
            vector = vector.flatten()
            dot_product = np.dot(query_vector, vector)
            query_norm = np.linalg.norm(query_vector)
            vector_norm = np.linalg.norm(vector)
            if query_norm == 0 or vector_norm == 0:
                similarity = 0.0
            else:
                similarity = dot_product / (query_norm * vector_norm)
            similarities.append((i, similarity))
        similarities.sort(key=lambda x: -float('inf')
                          if np.isnan(x[1]) else x[1], reverse=True)
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "content": self.contents[idx],
                "metadata": self.metadata[idx],
                "similarity": float(score)
            })
        return results


def create_embeddings(texts):
    if not texts:
        return []
    return embed_func(texts)


def process_document(pdf_path, chunk_size=1000, chunk_overlap=200, model="llava-hf/llava-1.5-7b-hf"):
    image_dir = os.path.join(DATA_DIR, "extracted_images")
    os.makedirs(image_dir, exist_ok=True)
    text_data, image_paths = extract_content_from_pdf(pdf_path, image_dir)
    chunked_text = chunk_text(text_data, chunk_size, chunk_overlap)
    image_data = process_images(image_paths, model)
    all_items = chunked_text + image_data
    contents = [item["content"] for item in all_items]
    logger.debug("Creating embeddings for all content...")
    embeddings = create_embeddings(contents)
    vector_store = MultiModalVectorStore()
    vector_store.add_items(all_items, embeddings)
    doc_info = {
        "text_count": len(chunked_text),
        "image_count": len(image_data),
        "total_items": len(all_items),
    }
    logger.debug(
        f"Added {len(all_items)} items to vector store ({len(chunked_text)} text chunks, {len(image_data)} image captions)")
    return vector_store, doc_info


def query_multimodal_rag(query, vector_store, k=5, model="llama-3.2-1b-instruct-4bit"):
    logger.debug(f"\n=== Processing query: {query} ===\n")
    query_embedding = create_embeddings(query)
    results = vector_store.similarity_search(query_embedding, k=k)
    text_results = [r for r in results if r["metadata"].get("type") == "text"]
    image_results = [
        r for r in results if r["metadata"].get("type") == "image"]
    logger.debug(
        f"Retrieved {len(results)} relevant items ({len(text_results)} text, {len(image_results)} image captions)")
    response = generate_response(query, results, model)
    return {
        "query": query,
        "results": results,
        "response": response,
        "text_results_count": len(text_results),
        "image_results_count": len(image_results)
    }


def generate_response(query, results, model="llama-3.2-1b-instruct-4bit"):
    context = ""
    for i, result in enumerate(results):
        content_type = "Text" if result["metadata"].get(
            "type") == "text" else "Image caption"
        page_num = result["metadata"].get("page", "unknown")
        context += f"[{content_type} from page {page_num}]\n"
        context += result["content"]
        context += "\n\n"
    system_prompt = "You are a helpful AI assistant. Answer the question based on the provided context. If the context is insufficient, acknowledge the limitation."
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"
    response = mlx.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model=model,
        temperature=0.1
    )
    return response["choices"][0]["message"]["content"]


def build_text_only_store(pdf_path, chunk_size=1000, chunk_overlap=200):
    text_data, _ = extract_content_from_pdf(pdf_path, None)
    chunked_text = chunk_text(text_data, chunk_size, chunk_overlap)
    contents = [item["content"] for item in chunked_text]
    logger.debug("Creating embeddings for text-only content...")
    embeddings = create_embeddings(contents)
    vector_store = MultiModalVectorStore()
    vector_store.add_items(chunked_text, embeddings)
    logger.debug(
        f"Added {len(chunked_text)} text items to text-only vector store")
    return vector_store


def evaluate_multimodal_vs_textonly(pdf_path, test_queries, reference_answers=None, model="llama-3.2-1b-instruct-4bit"):
    logger.debug("=== EVALUATING MULTI-MODAL RAG VS TEXT-ONLY RAG ===\n")
    logger.debug("\nProcessing document for multi-modal RAG...")
    mm_vector_store, mm_doc_info = process_document(pdf_path, model=model)
    logger.debug("\nProcessing document for text-only RAG...")
    text_vector_store = build_text_only_store(pdf_path)
    results = []
    for i, query in enumerate(test_queries):
        logger.debug(f"\n\n=== Evaluating Query {i+1}: {query} ===")
        reference = None
        if reference_answers and i < len(reference_answers):
            reference = reference_answers[i]
        logger.debug("\nRunning multi-modal RAG...")
        mm_result = query_multimodal_rag(query, mm_vector_store, model=model)
        logger.debug("\nRunning text-only RAG...")
        text_result = query_multimodal_rag(
            query, text_vector_store, model=model)
        comparison = compare_responses(
            query, mm_result["response"], text_result["response"], reference, model)
        results.append({
            "query": query,
            "multimodal_response": mm_result["response"],
            "textonly_response": text_result["response"],
            "multimodal_results": {
                "text_count": mm_result["text_results_count"],
                "image_count": mm_result["image_results_count"]
            },
            "reference_answer": reference,
            "comparison": comparison
        })
    overall_analysis = generate_overall_analysis(results, model)
    return {
        "results": results,
        "overall_analysis": overall_analysis,
        "multimodal_doc_info": mm_doc_info
    }


def compare_responses(query, mm_response, text_response, reference=None, model="llama-3.2-1b-instruct-4bit"):
    system_prompt = "You are an objective evaluator. Compare the two responses to the query and provide a concise evaluation. If a reference answer is provided, use it to assess accuracy and completeness."
    user_prompt = f"Query: {query}\n\nMulti-modal Response:\n{mm_response}\n\nText-only Response:\n{text_response}"
    if reference:
        user_prompt += f"\n\nReference Answer:\n{reference}"
    response = mlx.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model=model,
        temperature=0
    )
    return response["choices"][0]["message"]["content"]


def generate_overall_analysis(results, model="llama-3.2-1b-instruct-4bit"):
    evaluations_summary = ""
    for i, result in enumerate(results):
        evaluations_summary += f"Query {i+1}: {result['query']}\n"
        evaluations_summary += f"Multi-modal retrieved {result['multimodal_results']['text_count']} text chunks and {result['multimodal_results']['image_count']} image captions\n"
        evaluations_summary += f"Comparison summary: {result['comparison'][:200]}...\n\n"
    system_prompt = "Provide an overall analysis of the performance of multi-modal RAG versus text-only RAG based on the provided summaries."
    user_prompt = f"Evaluations Summary:\n{evaluations_summary}"
    response = mlx.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model=model,
        temperature=0
    )
    return response["choices"][0]["message"]["content"]


pdf_path = os.path.join(DATA_DIR, "attention_is_all_you_need.pdf")
test_queries = [
    "What is the BLEU score of the Transformer (base model)?",
]
reference_answers = [
    "The Transformer (base model) achieves a BLEU score of 27.3 on the WMT 2014 English-to-German translation task and 38.1 on the WMT 2014 English-to-French translation task.",
]
evaluation_results = evaluate_multimodal_vs_textonly(
    pdf_path=pdf_path,
    test_queries=test_queries,
    reference_answers=reference_answers
)
logger.debug("\n=== OVERALL ANALYSIS ===\n")
logger.debug(evaluation_results["overall_analysis"])
logger.info("\n\n[DONE]", bright=True)
