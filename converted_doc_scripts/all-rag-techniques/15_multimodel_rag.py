import base64
import json
import numpy as np
import os
import re
import shutil
import tempfile
from PIL import Image
from typing import List, Dict, Any
from jet.file.utils import save_file
from helpers import (
    setup_config, initialize_mlx, generate_embeddings,
    load_validation_data, generate_ai_response,
    load_json_data, SearchResult, SimpleVectorStore, DATA_DIR, DOCS_PATH
)


def chunk_text(text_data: List[Dict[str, Any]], chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
    """Chunk text data into overlapping segments."""
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


def encode_image(image_path: str) -> str:
    """Encode image to base64."""
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read())
        return encoded_image.decode('utf-8')


def generate_image_caption(image_path: str, mlx, model: str = "llava-hf/llava-1.5-7b-hf") -> str:
    """Generate caption for an image."""
    if not os.path.exists(image_path):
        return "Error: Image file not found"
    try:
        Image.open(image_path)
        base64_image = encode_image(image_path)
        response = mlx.chat(
            [
                {"role": "system", "content": "You are a helpful AI assistant."},
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


def process_images(image_paths: List[Dict[str, Any]], mlx, model: str = "llava-hf/llava-1.5-7b-hf") -> List[Dict[str, Any]]:
    """Process images to generate captions."""
    image_data = []
    logger.debug(f"Generating captions for {len(image_paths)} images...")
    for i, img_item in enumerate(image_paths):
        logger.debug(f"Processing image {i+1}/{len(image_paths)}...")
        img_path = img_item["path"]
        metadata = img_item["metadata"]
        caption = generate_image_caption(img_path, mlx, model)
        image_data.append({
            "content": caption,
            "metadata": metadata,
            "image_path": img_path
        })
    return image_data


def process_document(chunks: List[Dict[str, Any]], image_paths: List[Dict[str, Any]], embed_func, mlx, model: str = "llava-hf/llava-1.5-7b-hf") -> tuple[SimpleVectorStore, Dict[str, int]]:
    """Process document text and images into a vector store."""
    chunked_text = chunk_text(chunks)
    image_data = process_images(image_paths, mlx, model)
    all_items = chunked_text + image_data
    contents = [item["content"] for item in all_items]
    logger.debug("Creating embeddings for all content...")
    embeddings = generate_embeddings(contents, embed_func, logger)
    vector_store = SimpleVectorStore()
    for item, embedding in zip(all_items, embeddings):
        vector_store.add_item(
            text=item["content"],
            embedding=embedding,
            metadata=item.get("metadata", {})
        )
    doc_info = {
        "text_count": len(chunked_text),
        "image_count": len(image_data),
        "total_items": len(all_items),
    }
    logger.debug(
        f"Added {len(all_items)} items to vector store ({len(chunked_text)} text chunks, {len(image_data)} image captions)")
    return vector_store, doc_info


def query_multimodal_rag(query: str, vector_store: SimpleVectorStore, embed_func, mlx, k: int = 5, model: str = "llama-3.2-3b-instruct-4bit") -> Dict[str, Any]:
    """Run multimodal RAG query."""
    logger.debug(f"\n=== Processing query: {query} ===\n")
    query_embedding = embed_func(query)
    results = vector_store.search(query_embedding, top_k=k)
    text_results = [r for r in results if r["metadata"].get("type") == "text"]
    image_results = [
        r for r in results if r["metadata"].get("type") == "image"]
    logger.debug(
        f"Retrieved {len(results)} relevant items ({len(text_results)} text, {len(image_results)} image captions)")
    response = generate_ai_response(
        query,
        "You are a helpful AI assistant. Answer the question based on the provided context. If the context is insufficient, acknowledge the limitation.",
        results,
        mlx,
        logger,
        model=model
    )
    return {
        "query": query,
        "results": results,
        "response": response,
        "text_results_count": len(text_results),
        "image_results_count": len(image_results)
    }


def build_text_only_store(chunks: List[Dict[str, Any]], embed_func) -> SimpleVectorStore:
    """Build a text-only vector store."""
    chunked_text = chunk_text(chunks)
    contents = [item["content"] for item in chunked_text]
    logger.debug("Creating embeddings for text-only content...")
    embeddings = generate_embeddings(contents, embed_func, logger)
    vector_store = SimpleVectorStore()
    for item, embedding in zip(chunked_text, embeddings):
        vector_store.add_item(
            text=item["content"],
            embedding=embedding,
            metadata=item.get("metadata", {})
        )
    logger.debug(
        f"Added {len(chunked_text)} text items to text-only vector store")
    return vector_store


def evaluate_multimodal_vs_textonly(chunks: List[Dict[str, Any]], image_paths: List[Dict[str, Any]], test_queries: List[str], embed_func, mlx, reference_answers: List[str] = None, model: str = "llama-3.2-3b-instruct-4bit") -> Dict[str, Any]:
    """Evaluate multimodal RAG vs text-only RAG."""
    logger.debug("=== EVALUATING MULTI-MODAL RAG VS TEXT-ONLY RAG ===\n")
    logger.debug("\nProcessing document for multi-modal RAG...")
    mm_vector_store, mm_doc_info = process_document(
        chunks, image_paths, embed_func, mlx)
    logger.debug("\nProcessing document for text-only RAG...")
    text_vector_store = build_text_only_store(chunks, embed_func)
    results = []
    for i, query in enumerate(test_queries):
        logger.debug(f"\n\n=== Evaluating Query {i+1}: {query} ===")
        reference = None
        if reference_answers and i < len(reference_answers):
            reference = reference_answers[i]
        logger.debug("\nRunning multi-modal RAG...")
        mm_result = query_multimodal_rag(
            query, mm_vector_store, embed_func, mlx, model=model)
        logger.debug("\nRunning text-only RAG...")
        text_result = query_multimodal_rag(
            query, text_vector_store, embed_func, mlx, model=model)
        comparison = compare_responses(
            query, mm_result["response"], text_result["response"], reference, mlx, model)
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
    overall_analysis = generate_overall_analysis(results, mlx, model)
    return {
        "results": results,
        "overall_analysis": overall_analysis,
        "multimodal_doc_info": mm_doc_info
    }


def compare_responses(query: str, mm_response: str, text_response: str, reference: str = None, mlx=None, model: str = "llama-3.2-3b-instruct-4bit") -> str:
    """Compare multimodal and text-only responses."""
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


def generate_overall_analysis(results: List[Dict[str, Any]], mlx, model: str = "llama-3.2-3b-instruct-4bit") -> str:
    """Generate overall analysis of RAG approaches."""
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


script_dir, generated_dir, log_file, logger = setup_config(__file__)
mlx, embed_func = initialize_mlx(logger)
formatted_texts, original_chunks = load_json_data(DOCS_PATH, logger)
logger.info("Loaded pre-chunked data from DOCS_PATH")
# Assuming image paths are provided or extracted separately
image_paths = [
    {"path": os.path.join(DATA_DIR, f"image_{i}.png"), "metadata": {
        "source": "attention_is_all_you_need.pdf", "page": i+1, "image_index": i+1, "type": "image"}}
    for i in range(3)  # Example placeholder for image paths
]
test_queries = [
    "What is the BLEU score of the Transformer (base model)?",
]
reference_answers = [
    "The Transformer (base model) achieves a BLEU score of 27.3 on the WMT 2014 English-to-German translation task and 38.1 on the WMT 2014 English-to-French translation task.",
]
evaluation_results = evaluate_multimodal_vs_textonly(
    chunks=original_chunks,
    image_paths=image_paths,
    test_queries=test_queries,
    embed_func=embed_func,
    mlx=mlx,
    reference_answers=reference_answers
)
save_file(evaluation_results, f"{generated_dir}/evaluation_results.json")
logger.info(
    f"Saved evaluation results to {generated_dir}/evaluation_results.json")
logger.debug("\n=== OVERALL ANALYSIS ===\n")
logger.debug(evaluation_results["overall_analysis"])
logger.info("\n\n[DONE]", bright=True)
