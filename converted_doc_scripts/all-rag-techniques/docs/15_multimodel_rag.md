# Multi-Modal RAG with Image Captioning

In this notebook, I implement a Multi-Modal RAG system that extracts both text and images from documents, generates captions for images, and uses both content types to respond to queries. This approach enhances traditional RAG by incorporating visual information into the knowledge base.

Traditional RAG systems only work with text, but many documents contain crucial information in images, charts, and tables. By captioning these visual elements and incorporating them into our retrieval system, we can:

- Access information locked in figures and diagrams
- Understand tables and charts that complement the text
- Create a more comprehensive knowledge base
- Answer questions that rely on visual data

## Setting Up the Environment
We begin by importing necessary libraries.

```python
import os
import io
import numpy as np
import json
import fitz
from PIL import Image
from openai import OpenAI
import base64
import re
import tempfile
import shutil
```

## Setting Up the OpenAI API Client
We initialize the OpenAI client to generate embeddings and responses.

```python
# Initialize the OpenAI client with the base URL and API key
client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.getenv("OPENAI_API_KEY")  # Retrieve the API key from environment variables
)
```

## Document Processing Functions

```python
def extract_content_from_pdf(pdf_path, output_dir=None):
    """
    Extract both text and images from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file
        output_dir (str, optional): Directory to save extracted images

    Returns:
        Tuple[List[Dict], List[Dict]]: Text data and image data
    """
    # Create a temporary directory for images if not provided
    temp_dir = None
    if output_dir is None:
        temp_dir = tempfile.mkdtemp()
        output_dir = temp_dir
    else:
        os.makedirs(output_dir, exist_ok=True)

    text_data = []  # List to store extracted text data
    image_paths = []  # List to store paths of extracted images

    print(f"Extracting content from {pdf_path}...")

    try:
        with fitz.open(pdf_path) as pdf_file:
            # Loop through every page in the PDF
            for page_number in range(len(pdf_file)):
                page = pdf_file[page_number]

                # Extract text from the page
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

                # Extract images from the page
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    xref = img[0]  # XREF of the image
                    base_image = pdf_file.extract_image(xref)

                    if base_image:
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]

                        # Save the image to the output directory
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

        print(f"Extracted {len(text_data)} text segments and {len(image_paths)} images")
        return text_data, image_paths

    except Exception as e:
        print(f"Error extracting content: {e}")
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise
```

## Chunking Text Content

```python
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

        # Skip if text is too short
        if len(text) < chunk_size / 2:
            chunked_data.append({
                "content": text,
                "metadata": metadata
            })
            continue

        # Create chunks with overlap
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]  # Extract a chunk of the specified size
            if chunk:  # Ensure we don't add empty chunks
                chunks.append(chunk)

        # Add each chunk with updated metadata
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()  # Copy the original metadata
            chunk_metadata["chunk_index"] = i  # Add chunk index to metadata
            chunk_metadata["chunk_count"] = len(chunks)  # Add total chunk count to metadata

            chunked_data.append({
                "content": chunk,  # The chunk text
                "metadata": chunk_metadata  # The updated metadata
            })

    print(f"Created {len(chunked_data)} text chunks")  # Print the number of created chunks
    return chunked_data  # Return the list of chunked data
```

## Image Captioning with OpenAI Vision

```python
def encode_image(image_path):
    """
    Encode an image file as base64.

    Args:
        image_path (str): Path to the image file

    Returns:
        str: Base64 encoded image
    """
    # Open the image file in binary read mode
    with open(image_path, "rb") as image_file:
        # Read the image file and encode it to base64
        encoded_image = base64.b64encode(image_file.read())
        # Decode the base64 bytes to a string and return
        return encoded_image.decode('utf-8')
```

```python
def generate_image_caption(image_path):
    """
    Generate a caption for an image using OpenAI's vision capabilities.

    Args:
        image_path (str): Path to the image file

    Returns:
        str: Generated caption
    """
    # Check if the file exists and is an image
    if not os.path.exists(image_path):
        return "Error: Image file not found"

    try:
        # Open and validate the image
        Image.open(image_path)

        # Encode the image to base64
        base64_image = encode_image(image_path)

        # Create the API request to generate the caption
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

        # Extract the caption from the response
        caption = response.choices[0].message.content
        return caption

    except Exception as e:
        # Return an error message if an exception occurs
        return f"Error generating caption: {str(e)}"
```

```python
def process_images(image_paths):
    """
    Process all images and generate captions.

    Args:
        image_paths (List[Dict]): Paths to extracted images

    Returns:
        List[Dict]: Image data with captions
    """
    image_data = []  # Initialize an empty list to store image data with captions

    print(f"Generating captions for {len(image_paths)} images...")  # Print the number of images to process
    for i, img_item in enumerate(image_paths):
        print(f"Processing image {i+1}/{len(image_paths)}...")  # Print the current image being processed
        img_path = img_item["path"]  # Get the image path
        metadata = img_item["metadata"]  # Get the image metadata

        # Generate caption for the image
        caption = generate_image_caption(img_path)

        # Add the image data with caption to the list
        image_data.append({
            "content": caption,  # The generated caption
            "metadata": metadata,  # The image metadata
            "image_path": img_path  # The path to the image
        })

    return image_data  # Return the list of image data with captions
```

## Simple Vector Store Implementation

```python
class MultiModalVectorStore:
    """
    A simple vector store implementation for multi-modal content.
    """
    def __init__(self):
        # Initialize lists to store vectors, contents, and metadata
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
        # Append the embedding vector, content, and metadata to their respective lists
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
        # Loop through items and embeddings and add each to the vector store
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
        # Return an empty list if there are no vectors in the store
        if not self.vectors:
            return []

        # Convert query embedding to numpy array
        query_vector = np.array(query_embedding)

        # Calculate similarities using cosine similarity
        similarities = []
        for i, vector in enumerate(self.vectors):
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities.append((i, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Return top k results
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "content": self.contents[idx],
                "metadata": self.metadata[idx],
                "similarity": float(score)  # Convert to float for JSON serialization
            })

        return results
```

## Creating Embeddings

```python
def create_embeddings(texts, model="BAAI/bge-en-icl"):
    """
    Create embeddings for the given texts.

    Args:
        texts (List[str]): Input texts
        model (str): Embedding model name

    Returns:
        List[List[float]]: Embedding vectors
    """
    # Handle empty input
    if not texts:
        return []

    # Process in batches if needed (OpenAI API limits)
    batch_size = 100
    all_embeddings = []

    # Iterate over the input texts in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]  # Get the current batch of texts

        # Create embeddings for the current batch
        response = client.embeddings.create(
            model=model,
            input=batch
        )

        # Extract embeddings from the response
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)  # Add the batch embeddings to the list

    return all_embeddings  # Return all embeddings
```

## Complete Processing Pipeline

```python
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
    # Create a directory for extracted images
    image_dir = "extracted_images"
    os.makedirs(image_dir, exist_ok=True)

    # Extract text and images from the PDF
    text_data, image_paths = extract_content_from_pdf(pdf_path, image_dir)

    # Chunk the extracted text
    chunked_text = chunk_text(text_data, chunk_size, chunk_overlap)

    # Process the extracted images to generate captions
    image_data = process_images(image_paths)

    # Combine all content items (text chunks and image captions)
    all_items = chunked_text + image_data

    # Extract content for embedding
    contents = [item["content"] for item in all_items]

    # Create embeddings for all content
    print("Creating embeddings for all content...")
    embeddings = create_embeddings(contents)

    # Build the vector store and add items with their embeddings
    vector_store = MultiModalVectorStore()
    vector_store.add_items(all_items, embeddings)

    # Prepare document info with counts of text chunks and image captions
    doc_info = {
        "text_count": len(chunked_text),
        "image_count": len(image_data),
        "total_items": len(all_items),
    }

    # Print summary of added items
    print(f"Added {len(all_items)} items to vector store ({len(chunked_text)} text chunks, {len(image_data)} image captions)")

    # Return the vector store and document info
    return vector_store, doc_info
```

## Query Processing and Response Generation

```python
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
    print(f"\n=== Processing query: {query} ===\n")

    # Generate embedding for the query
    query_embedding = create_embeddings(query)

    # Retrieve relevant content from the vector store
    results = vector_store.similarity_search(query_embedding, k=k)

    # Separate text and image results
    text_results = [r for r in results if r["metadata"].get("type") == "text"]
    image_results = [r for r in results if r["metadata"].get("type") == "image"]

    print(f"Retrieved {len(results)} relevant items ({len(text_results)} text, {len(image_results)} image captions)")

    # Generate a response using the retrieved content
    response = generate_response(query, results)

    return {
        "query": query,
        "results": results,
        "response": response,
        "text_results_count": len(text_results),
        "image_results_count": len(image_results)
    }
```

```python

def generate_response(query, results):
    """
    Generate a response based on the query and retrieved results.

    Args:
        query (str): User query
        results (List[Dict]): Retrieved content

    Returns:
        str: Generated response
    """
    # Format the context from the retrieved results
    context = ""

    for i, result in enumerate(results):
        # Determine the type of content (text or image caption)
        content_type = "Text" if result["metadata"].get("type") == "text" else "Image caption"
        # Get the page number from the metadata
        page_num = result["metadata"].get("page", "unknown")

        # Append the content type and page number to the context
        context += f"[{content_type} from page {page_num}]\n"
        # Append the actual content to the context
        context += result["content"]
        context += "\n\n"

    # System message to guide the AI assistant
    system_message = """You are an AI assistant specializing in answering questions about documents
    that contain both text and images. You have been given relevant text passages and image captions
    from the document. Use this information to provide a comprehensive, accurate response to the query.
    If information comes from an image or chart, mention this in your answer.
    If the retrieved information doesn't fully answer the query, acknowledge the limitations."""

    # User message containing the query and the formatted context
    user_message = f"""Query: {query}

    Retrieved content:
    {context}

    Please answer the query based on the retrieved content.
    """

    # Generate the response using the OpenAI API
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        temperature=0.1
    )

    # Return the generated response
    return response.choices[0].message.content
```

## Evaluation Against Text-Only RAG

```python
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
    # Extract text from PDF (reuse function but ignore images)
    text_data, _ = extract_content_from_pdf(pdf_path, None)

    # Chunk text
    chunked_text = chunk_text(text_data, chunk_size, chunk_overlap)

    # Extract content for embedding
    contents = [item["content"] for item in chunked_text]

    # Create embeddings
    print("Creating embeddings for text-only content...")
    embeddings = create_embeddings(contents)

    # Build vector store
    vector_store = MultiModalVectorStore()
    vector_store.add_items(chunked_text, embeddings)

    print(f"Added {len(chunked_text)} text items to text-only vector store")
    return vector_store
```

```python
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
    print("=== EVALUATING MULTI-MODAL RAG VS TEXT-ONLY RAG ===\n")

    # Process document for multi-modal RAG
    print("\nProcessing document for multi-modal RAG...")
    mm_vector_store, mm_doc_info = process_document(pdf_path)

    # Build text-only store
    print("\nProcessing document for text-only RAG...")
    text_vector_store = build_text_only_store(pdf_path)

    # Run evaluation for each query
    results = []

    for i, query in enumerate(test_queries):
        print(f"\n\n=== Evaluating Query {i+1}: {query} ===")

        # Get reference answer if available
        reference = None
        if reference_answers and i < len(reference_answers):
            reference = reference_answers[i]

        # Run multi-modal RAG
        print("\nRunning multi-modal RAG...")
        mm_result = query_multimodal_rag(query, mm_vector_store)

        # Run text-only RAG
        print("\nRunning text-only RAG...")
        text_result = query_multimodal_rag(query, text_vector_store)

        # Compare responses
        comparison = compare_responses(query, mm_result["response"], text_result["response"], reference)

        # Add to results
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

    # Generate overall analysis
    overall_analysis = generate_overall_analysis(results)

    return {
        "results": results,
        "overall_analysis": overall_analysis,
        "multimodal_doc_info": mm_doc_info
    }
```

```python
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
    # System prompt for the evaluator
    system_prompt = """You are an expert evaluator comparing two RAG systems:
    1. Multi-modal RAG: Retrieves from both text and image captions
    2. Text-only RAG: Retrieves only from text

    Evaluate which response better answers the query based on:
    - Accuracy and correctness
    - Completeness of information
    - Relevance to the query
    - Unique information from visual elements (for multi-modal)"""

    # User prompt with query and responses
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

    # Generate comparison using meta-llama/Llama-3.2-3B-Instruct
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content
```

```python
def generate_overall_analysis(results):
    """
    Generate an overall analysis of multi-modal vs text-only RAG.

    Args:
        results (List[Dict]): Evaluation results for each query

    Returns:
        str: Overall analysis
    """
    # System prompt for the evaluator
    system_prompt = """You are an expert evaluator of RAG systems. Provide an overall analysis comparing
    multi-modal RAG (text + images) versus text-only RAG based on multiple test queries.

    Focus on:
    1. Types of queries where multi-modal RAG outperforms text-only
    2. Specific advantages of incorporating image information
    3. Any disadvantages or limitations of the multi-modal approach
    4. Overall recommendation on when to use each approach"""

    # Create summary of evaluations
    evaluations_summary = ""
    for i, result in enumerate(results):
        evaluations_summary += f"Query {i+1}: {result['query']}\n"
        evaluations_summary += f"Multi-modal retrieved {result['multimodal_results']['text_count']} text chunks and {result['multimodal_results']['image_count']} image captions\n"
        evaluations_summary += f"Comparison summary: {result['comparison'][:200]}...\n\n"

    # User prompt with evaluations summary
    user_prompt = f"""Based on the following evaluations of multi-modal vs text-only RAG across {len(results)} queries,
    provide an overall analysis comparing these two approaches:

    {evaluations_summary}

    Please provide a comprehensive analysis of the relative strengths and weaknesses of multi-modal RAG
    compared to text-only RAG, with specific attention to how image information contributed (or didn't contribute) to response quality."""

    # Generate overall analysis using meta-llama/Llama-3.2-3B-Instruct
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content
```

## Evaluation on Multi-Modal RAG vs Text-Only RAG

```python
# Path to your PDF document
pdf_path = "data/attention_is_all_you_need.pdf"

# Define test queries targeting both text and visual content
test_queries = [
    "What is the BLEU score of the Transformer (base model)?",
]

# Optional reference answers for evaluation
reference_answers = [
    "The Transformer (base model) achieves a BLEU score of 27.3 on the WMT 2014 English-to-German translation task and 38.1 on the WMT 2014 English-to-French translation task.",
]

# Run evaluation
evaluation_results = evaluate_multimodal_vs_textonly(
    pdf_path=pdf_path,
    test_queries=test_queries,
    reference_answers=reference_answers
)

# Print overall analysis
print("\n=== OVERALL ANALYSIS ===\n")
print(evaluation_results["overall_analysis"])
```

```output
=== EVALUATING MULTI-MODAL RAG VS TEXT-ONLY RAG ===


Processing document for multi-modal RAG...
Extracting content from data/attention_is_all_you_need.pdf...
```

```output
Extracted 15 text segments and 3 images
Created 59 text chunks
Generating captions for 3 images...
Processing image 1/3...
Processing image 2/3...
Processing image 3/3...
Creating embeddings for all content...
Added 62 items to vector store (59 text chunks, 3 image captions)

Processing document for text-only RAG...
Extracting content from data/attention_is_all_you_need.pdf...
Extracted 15 text segments and 3 images
Created 59 text chunks
Creating embeddings for text-only content...
Added 59 text items to text-only vector store


=== Evaluating Query 1: What is the BLEU score of the Transformer (base model)? ===

Running multi-modal RAG...

=== Processing query: What is the BLEU score of the Transformer (base model)? ===

```

```output
C:\Users\faree\AppData\Local\Temp\ipykernel_14692\2117883450.py:75: DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)
  "similarity": float(score)  # Convert to float for JSON serialization
```

```output
Retrieved 5 relevant items (5 text, 0 image captions)

Running text-only RAG...

=== Processing query: What is the BLEU score of the Transformer (base model)? ===

Retrieved 5 relevant items (5 text, 0 image captions)

=== OVERALL ANALYSIS ===

**Overall Analysis: Multi-Modal RAG vs Text-Only RAG**

Our analysis compares the performance of multi-modal RAG (text + images) and text-only RAG across multiple test queries. We evaluate the strengths and weaknesses of each approach, focusing on the types of queries where multi-modal RAG outperforms text-only, the advantages of incorporating image information, and the limitations of the multi-modal approach.

**Advantages of Multi-Modal RAG**

1. **Improved Contextual Understanding**: Multi-modal RAG can leverage both text and image information to better understand the context of a query. This can lead to more accurate and informative responses, especially when the query requires a deeper understanding of the topic.
2. **Enhanced Visual Cues**: Images can provide visual cues that can help disambiguate ambiguous queries or provide additional context that is not explicitly stated in the text. For example, in Query 1, the image of the Transformer model could provide visual confirmation of the query, which was not present in the text-only response.
3. **Increased Retrieval Precision**: Multi-modal RAG can retrieve more relevant text chunks and image captions, leading to more accurate and precise responses. In Query 1, the multi-modal RAG retrieved 5 text chunks and 0 image captions, which may not have been sufficient to provide a clear answer.

**Disadvantages of Multi-Modal RAG**

1. **Increased Complexity**: Multi-modal RAG requires more complex processing and retrieval mechanisms, which can increase the computational cost and make the system more difficult to train and deploy.
2. **Image Quality and Relevance**: The quality and relevance of the image captions can significantly impact the performance of the multi-modal RAG. Poor-quality or irrelevant images can lead to suboptimal responses.
3. **Overreliance on Image Information**: If the image information is not relevant or accurate, the multi-modal RAG may rely too heavily on it, leading to suboptimal responses.

**Types of Queries where Multi-Modal RAG Outperforms Text-Only**

1. **Visual-Spatial Queries**: Multi-modal RAG can perform better on visual-spatial queries that require a deeper understanding of the visual context, such as Query 1 (What is the BLEU score of the Transformer (base model)?).
2. **Ambiguous Queries**: Multi-modal RAG can help disambiguate ambiguous queries by leveraging both text and image information, leading to more accurate and informative responses.
3. **Multi-Modal Queries**: Multi-modal RAG can perform better on multi-modal queries that require the integration of both text and image information, such as Query 1.

**Specific Advantages of Incorporating Image Information**

1. **Visual Confirmation**: Images can provide visual confirmation of the query, which can help disambiguate ambiguous queries or provide additional context that is not explicitly stated in the text.
2. **Visual Cues**: Images can provide visual cues that can help the model understand the context of the query, leading to more accurate and informative responses.
3. **Multimodal Fusion**: Images can be fused with text information to provide a more comprehensive understanding of the query, leading to more accurate and informative responses.

**Overall Recommendation**

1. **Use Text-Only RAG for Simple Queries**: Text-only RAG is sufficient for simple queries that do not require a deep understanding of the context or visual information.
2. **Use Multi-Modal RAG for Complex Queries**: Multi-modal RAG is recommended for complex queries that require a deeper understanding of the context, visual information, or both.
3. **Use Multi-Modal RAG for Ambiguous Queries**: Multi-modal RAG can help disambiguate ambiguous queries by leveraging both text and image information, leading to more accurate and informative responses.

In conclusion, multi-modal RAG can outperform text-only RAG in certain types of queries, particularly those that require a deeper understanding of the context, visual information, or both. However, the multi-modal approach also has its limitations, such as increased complexity and the potential for overreliance on image information. The choice of approach depends on the specific use case and the type of queries being addressed.
```
