from PIL import Image
from jet.logger import CustomLogger
from openai import Ollama
import base64
import fitz
import io
import json
import numpy as np
import os
import re
import shutil
import tempfile


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

file_name = os.path.splitext(os.path.basename(__file__))[0]
GENERATED_DIR = os.path.join("results", file_name)
os.makedirs(GENERATED_DIR, exist_ok=True)

"""
# Multi-Modal RAG with Image Captioning

In this notebook, I implement a Multi-Modal RAG system that extracts both text and images from documents, generates captions for images, and uses both content types to respond to queries. This approach enhances traditional RAG by incorporating visual information into the knowledge base.

Traditional RAG systems only work with text, but many documents contain crucial information in images, charts, and tables. By captioning these visual elements and incorporating them into our retrieval system, we can:

- Access information locked in figures and diagrams
- Understand tables and charts that complement the text
- Create a more comprehensive knowledge base
- Answer questions that rely on visual data

## Setting Up the Environment
We begin by importing necessary libraries.
"""
logger.info("# Multi-Modal RAG with Image Captioning")


"""
## Setting Up the Ollama API Client
We initialize the Ollama client to generate embeddings and responses.
"""
logger.info("## Setting Up the Ollama API Client")

client = Ollama(
    base_url="https://api.studio.nebius.com/v1/",
#     api_key=os.getenv("OPENAI_API_KEY")  # Retrieve the API key from environment variables
)

"""
## Document Processing Functions
"""
logger.info("## Document Processing Functions")

def extract_content_from_pdf(pdf_path, output_dir=None):
    """
    Extract both text and images from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file
        output_dir (str, optional): Directory to save extracted images

    Returns:
        Tuple[List[Dict], List[Dict]]: Text data and image data
    """
    temp_dir = None
    if output_dir is None:
        temp_dir = tempfile.mkdtemp()
        output_dir = temp_dir
    else:
        os.makedirs(output_dir, exist_ok=True)

    text_data = []  # List to store extracted text data
    image_paths = []  # List to store paths of extracted images

    logger.debug(f"Extracting content from {pdf_path}...")

    try:
        with fitz.open(pdf_path) as pdf_file:
            for page_number in range(len(pdf_file)):
                page = pdf_file[page_number]

                text = page.get_text().strip()
                if text:
                    text_data.append({
                        "content": text,
                        "metadata": {
                            "source": pdf_path,
                            "page": page_number + 1,
                            "type": "text"
                        }
                    })

                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    xref = img[0]  # XREF of the image
                    base_image = pdf_file.extract_image(xref)

                    if base_image:
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]

                        img_filename = f"page_{page_number+1}_img_{img_index+1}.{image_ext}"
                        img_path = os.path.join(output_dir, img_filename)

                        with open(img_path, "wb") as img_file:
                            img_file.write(image_bytes)

                        image_paths.append({
                            "path": img_path,
                            "metadata": {
                                "source": pdf_path,
                                "page": page_number + 1,
                                "image_index": img_index + 1,
                                "type": "image"
                            }
                        })

        logger.debug(f"Extracted {len(text_data)} text segments and {len(image_paths)} images")
        return text_data, image_paths

    except Exception as e:
        logger.debug(f"Error extracting content: {e}")
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise

"""
## Chunking Text Content
"""
logger.info("## Chunking Text Content")

def chunk_text(text_data, chunk_size=1000, overlap=200):
    """
    Split text data into overlapping chunks.

    Args:
        text_data (List[Dict]): Text data extracted from PDF
        chunk_size (int): Size of each chunk in characters
        overlap (int): Overlap between chunks in characters

    Returns:
        List[Dict]: Chunked text data
    """
    chunked_data = []  # Initialize an empty list to store chunked data

    for item in text_data:
        text = item["content"]  # Extract the text content
        metadata = item["metadata"]  # Extract the metadata

        if len(text) < chunk_size / 2:
            chunked_data.append({
                "content": text,
                "metadata": metadata
            })
            continue

        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]  # Extract a chunk of the specified size
            if chunk:  # Ensure we don't add empty chunks
                chunks.append(chunk)

        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()  # Copy the original metadata
            chunk_metadata["chunk_index"] = i  # Add chunk index to metadata
            chunk_metadata["chunk_count"] = len(chunks)  # Add total chunk count to metadata

            chunked_data.append({
                "content": chunk,  # The chunk text
                "metadata": chunk_metadata  # The updated metadata
            })

    logger.debug(f"Created {len(chunked_data)} text chunks")  # Print the number of created chunks
    return chunked_data  # Return the list of chunked data

"""
## Image Captioning with Ollama Vision
"""
logger.info("## Image Captioning with Ollama Vision")

def encode_image(image_path):
    """
    Encode an image file as base64.

    Args:
        image_path (str): Path to the image file

    Returns:
        str: Base64 encoded image
    """
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read())
        return encoded_image.decode('utf-8')

def generate_image_caption(image_path):
    """
    Generate a caption for an image using Ollama's vision capabilities.

    Args:
        image_path (str): Path to the image file

    Returns:
        str: Generated caption
    """
    if not os.path.exists(image_path):
        return "Error: Image file not found"

    try:
        Image.open(image_path)

        base64_image = encode_image(image_path)

        response = client.chat.completions.create(
            model="llava-hf/llava-1.5-7b-hf", # Use the llava-1.5-7b model
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant specialized in describing images from academic papers. "
                    "Provide detailed captions for the image that capture key information. "
                    "If the image contains charts, tables, or diagrams, describe their content and purpose clearly. "
                    "Your caption should be optimized for future retrieval when people ask questions about this content."
                },
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
            max_tokens=300
        )

        caption = response.choices[0].message.content
        return caption

    except Exception as e:
        return f"Error generating caption: {str(e)}"

def process_images(image_paths):
    """
    Process all images and generate captions.

    Args:
        image_paths (List[Dict]): Paths to extracted images

    Returns:
        List[Dict]: Image data with captions
    """
    image_data = []  # Initialize an empty list to store image data with captions

    logger.debug(f"Generating captions for {len(image_paths)} images...")  # Print the number of images to process
    for i, img_item in enumerate(image_paths):
        logger.debug(f"Processing image {i+1}/{len(image_paths)}...")  # Print the current image being processed
        img_path = img_item["path"]  # Get the image path
        metadata = img_item["metadata"]  # Get the image metadata

        caption = generate_image_caption(img_path)

        image_data.append({
            "content": caption,  # The generated caption
            "metadata": metadata,  # The image metadata
            "image_path": img_path  # The path to the image
        })

    return image_data  # Return the list of image data with captions

"""
## Simple Vector Store Implementation
"""
logger.info("## Simple Vector Store Implementation")

class MultiModalVectorStore:
    """
    A simple vector store implementation for multi-modal content.
    """
    def __init__(self):
        self.vectors = []
        self.contents = []
        self.metadata = []

    def add_item(self, content, embedding, metadata=None):
        """
        Add an item to the vector store.

        Args:
            content (str): The content (text or image caption)
            embedding (List[float]): The embedding vector
            metadata (Dict, optional): Additional metadata
        """
        self.vectors.append(np.array(embedding))
        self.contents.append(content)
        self.metadata.append(metadata or {})

    def add_items(self, items, embeddings):
        """
        Add multiple items to the vector store.

        Args:
            items (List[Dict]): List of content items
            embeddings (List[List[float]]): List of embedding vectors
        """
        for item, embedding in zip(items, embeddings):
            self.add_item(
                content=item["content"],
                embedding=embedding,
                metadata=item.get("metadata", {})
            )

    def similarity_search(self, query_embedding, k=5):
        """
        Find the most similar items to a query embedding.

        Args:
            query_embedding (List[float]): Query embedding vector
            k (int): Number of results to return

        Returns:
            List[Dict]: Top k most similar items
        """
        if not self.vectors:
            return []

        query_vector = np.array(query_embedding)

        similarities = []
        for i, vector in enumerate(self.vectors):
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities.append((i, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)

        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "content": self.contents[idx],
                "metadata": self.metadata[idx],
                "similarity": float(score)  # Convert to float for JSON serialization
            })

        return results

"""
## Creating Embeddings
"""
logger.info("## Creating Embeddings")

def create_embeddings(texts, model="BAAI/bge-en-icl"):
    """
    Create embeddings for the given texts.

    Args:
        texts (List[str]): Input texts
        model (str): Embedding model name

    Returns:
        List[List[float]]: Embedding vectors
    """
    if not texts:
        return []

    batch_size = 100
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]  # Get the current batch of texts

        response = client.embeddings.create(
            model=model,
            input=batch
        )

        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)  # Add the batch embeddings to the list

    return all_embeddings  # Return all embeddings

"""
## Complete Processing Pipeline
"""
logger.info("## Complete Processing Pipeline")

def process_document(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    Process a document for multi-modal RAG.

    Args:
        pdf_path (str): Path to the PDF file
        chunk_size (int): Size of each chunk in characters
        chunk_overlap (int): Overlap between chunks in characters

    Returns:
        Tuple[MultiModalVectorStore, Dict]: Vector store and document info
    """
    image_dir = "extracted_images"
    os.makedirs(image_dir, exist_ok=True)

    text_data, image_paths = extract_content_from_pdf(pdf_path, image_dir)

    chunked_text = chunk_text(text_data, chunk_size, chunk_overlap)

    image_data = process_images(image_paths)

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

    logger.debug(f"Added {len(all_items)} items to vector store ({len(chunked_text)} text chunks, {len(image_data)} image captions)")

    return vector_store, doc_info

"""
## Query Processing and Response Generation
"""
logger.info("## Query Processing and Response Generation")

def query_multimodal_rag(query, vector_store, k=5):
    """
    Query the multi-modal RAG system.

    Args:
        query (str): User query
        vector_store (MultiModalVectorStore): Vector store with document content
        k (int): Number of results to retrieve

    Returns:
        Dict: Query results and generated response
    """
    logger.debug(f"\n=== Processing query: {query} ===\n")

    query_embedding = create_embeddings(query)

    results = vector_store.similarity_search(query_embedding, k=k)

    text_results = [r for r in results if r["metadata"].get("type") == "text"]
    image_results = [r for r in results if r["metadata"].get("type") == "image"]

    logger.debug(f"Retrieved {len(results)} relevant items ({len(text_results)} text, {len(image_results)} image captions)")

    response = generate_response(query, results)

    return {
        "query": query,
        "results": results,
        "response": response,
        "text_results_count": len(text_results),
        "image_results_count": len(image_results)
    }

def generate_response(query, results):
    """
    Generate a response based on the query and retrieved results.

    Args:
        query (str): User query
        results (List[Dict]): Retrieved content

    Returns:
        str: Generated response
    """
    context = ""

    for i, result in enumerate(results):
        content_type = "Text" if result["metadata"].get("type") == "text" else "Image caption"
        page_num = result["metadata"].get("page", "unknown")

        context += f"[{content_type} from page {page_num}]\n"
        context += result["content"]
        context += "\n\n"

    system_message = """You are an AI assistant specializing in answering questions about documents
    that contain both text and images. You have been given relevant text passages and image captions
    from the document. Use this information to provide a comprehensive, accurate response to the query.
    If information comes from an image or chart, mention this in your answer.
    If the retrieved information doesn't fully answer the query, acknowledge the limitations."""

    user_message = f"""Query: {query}

    Retrieved content:
    {context}

    Please answer the query based on the retrieved content.
    """

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        temperature=0.1
    )

    return response.choices[0].message.content

"""
## Evaluation Against Text-Only RAG
"""
logger.info("## Evaluation Against Text-Only RAG")

def build_text_only_store(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    Build a text-only vector store for comparison.

    Args:
        pdf_path (str): Path to the PDF file
        chunk_size (int): Size of each chunk in characters
        chunk_overlap (int): Overlap between chunks in characters

    Returns:
        MultiModalVectorStore: Text-only vector store
    """
    text_data, _ = extract_content_from_pdf(pdf_path, None)

    chunked_text = chunk_text(text_data, chunk_size, chunk_overlap)

    contents = [item["content"] for item in chunked_text]

    logger.debug("Creating embeddings for text-only content...")
    embeddings = create_embeddings(contents)

    vector_store = MultiModalVectorStore()
    vector_store.add_items(chunked_text, embeddings)

    logger.debug(f"Added {len(chunked_text)} text items to text-only vector store")
    return vector_store

def evaluate_multimodal_vs_textonly(pdf_path, test_queries, reference_answers=None):
    """
    Compare multi-modal RAG with text-only RAG.

    Args:
        pdf_path (str): Path to the PDF file
        test_queries (List[str]): Test queries
        reference_answers (List[str], optional): Reference answers

    Returns:
        Dict: Evaluation results
    """
    logger.debug("=== EVALUATING MULTI-MODAL RAG VS TEXT-ONLY RAG ===\n")

    logger.debug("\nProcessing document for multi-modal RAG...")
    mm_vector_store, mm_doc_info = process_document(pdf_path)

    logger.debug("\nProcessing document for text-only RAG...")
    text_vector_store = build_text_only_store(pdf_path)

    results = []

    for i, query in enumerate(test_queries):
        logger.debug(f"\n\n=== Evaluating Query {i+1}: {query} ===")

        reference = None
        if reference_answers and i < len(reference_answers):
            reference = reference_answers[i]

        logger.debug("\nRunning multi-modal RAG...")
        mm_result = query_multimodal_rag(query, mm_vector_store)

        logger.debug("\nRunning text-only RAG...")
        text_result = query_multimodal_rag(query, text_vector_store)

        comparison = compare_responses(query, mm_result["response"], text_result["response"], reference)

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

    overall_analysis = generate_overall_analysis(results)

    return {
        "results": results,
        "overall_analysis": overall_analysis,
        "multimodal_doc_info": mm_doc_info
    }

def compare_responses(query, mm_response, text_response, reference=None):
    """
    Compare multi-modal and text-only responses.

    Args:
        query (str): User query
        mm_response (str): Multi-modal response
        text_response (str): Text-only response
        reference (str, optional): Reference answer

    Returns:
        str: Comparison analysis
    """
    system_prompt = """You are an expert evaluator comparing two RAG systems:
    1. Multi-modal RAG: Retrieves from both text and image captions
    2. Text-only RAG: Retrieves only from text

    Evaluate which response better answers the query based on:
    - Accuracy and correctness
    - Completeness of information
    - Relevance to the query
    - Unique information from visual elements (for multi-modal)"""

    user_prompt = f"""Query: {query}

    Multi-modal RAG Response:
    {mm_response}

    Text-only RAG Response:
    {text_response}
    """

    if reference:
        user_prompt += f"""
    Reference Answer:
    {reference}
    """

        user_prompt += """
    Compare these responses and explain which one better answers the query and why.
    Note any specific information that came from images in the multi-modal response.
    """

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content

def generate_overall_analysis(results):
    """
    Generate an overall analysis of multi-modal vs text-only RAG.

    Args:
        results (List[Dict]): Evaluation results for each query

    Returns:
        str: Overall analysis
    """
    system_prompt = """You are an expert evaluator of RAG systems. Provide an overall analysis comparing
    multi-modal RAG (text + images) versus text-only RAG based on multiple test queries.

    Focus on:
    1. Types of queries where multi-modal RAG outperforms text-only
    2. Specific advantages of incorporating image information
    3. Any disadvantages or limitations of the multi-modal approach
    4. Overall recommendation on when to use each approach"""

    evaluations_summary = ""
    for i, result in enumerate(results):
        evaluations_summary += f"Query {i+1}: {result['query']}\n"
        evaluations_summary += f"Multi-modal retrieved {result['multimodal_results']['text_count']} text chunks and {result['multimodal_results']['image_count']} image captions\n"
        evaluations_summary += f"Comparison summary: {result['comparison'][:200]}...\n\n"

    user_prompt = f"""Based on the following evaluations of multi-modal vs text-only RAG across {len(results)} queries,
    provide an overall analysis comparing these two approaches:

    {evaluations_summary}

    Please provide a comprehensive analysis of the relative strengths and weaknesses of multi-modal RAG
    compared to text-only RAG, with specific attention to how image information contributed (or didn't contribute) to response quality."""

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content

"""
## Evaluation on Multi-Modal RAG vs Text-Only RAG
"""
logger.info("## Evaluation on Multi-Modal RAG vs Text-Only RAG")

pdf_path = f"{GENERATED_DIR}/attention_is_all_you_need.pdf"

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