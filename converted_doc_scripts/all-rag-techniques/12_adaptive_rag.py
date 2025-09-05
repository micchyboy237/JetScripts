import numpy as np
import re
from typing import List, Dict, Any
from jet.file.utils import save_file
from helpers import (
    setup_config, initialize_mlx, generate_embeddings,
    load_validation_data, generate_ai_response,
    load_json_data, SearchResult, SimpleVectorStore, DATA_DIR, DOCS_PATH
)
from jet.llm.mlx.remote import generation as gen


def chunk_text(text: str, n: int, overlap: int) -> List[str]:
    """Chunk text into overlapping segments."""
    chunks = []
    for i in range(0, len(text), n - overlap):
        chunks.append(text[i:i + n])
    return chunks


def process_document(chunks: List[Dict[str, Any]], embed_func) -> tuple[List[str], SimpleVectorStore]:
    """Process document chunks and store in vector store."""
    logger.debug("Processing chunks...")
    text_chunks = [chunk["text"] for chunk in chunks]
    logger.debug(f"Created {len(text_chunks)} text chunks")
    chunk_embeddings = generate_embeddings(text_chunks, embed_func, logger)
    store = SimpleVectorStore()
    for i, (chunk, embedding) in enumerate(zip(text_chunks, chunk_embeddings)):
        store.add_item(
            text=chunk,
            embedding=embedding,
            metadata={"index": i, "source": chunks[i]["metadata"]["doc_index"]}
        )
    logger.debug(f"Added {len(text_chunks)} chunks to the vector store")
    return text_chunks, store


def classify_query(query: str, mlx, model=None) -> str:
    """Classify the query type."""
    system_prompt = "Classify the query as Factual, Analytical, Opinion, or Contextual. Respond with only the category name."
    user_prompt = f"Query: {query}"
    response = ""
    for chunk in gen.stream_chat(
        messages=[
            {"role": "user", "content": user_prompt}
        ],
        system_prompt=system_prompt,
        model=model,
        max_tokens=512,
        temperature=0
    ):
        content = chunk["choices"][0]["message"]["content"]
        response += content
        logger.success(content, flush=True)
    category = response.strip()
    valid_categories = ["Factual", "Analytical", "Opinion", "Contextual"]
    return category if category in valid_categories else "Factual"


def factual_retrieval_strategy(query: str, vector_store: SimpleVectorStore, embed_func, mlx, k: int = 4, model=None) -> List[Dict[str, Any]]:
    """Factual retrieval strategy with query enhancement."""
    logger.debug(f"Executing Factual retrieval strategy for: '{query}'")
    system_prompt = "Enhance this factual query to improve retrieval accuracy. Respond with only the enhanced query."
    user_prompt = f"Query: {query}"
    response = ""
    for chunk in gen.stream_chat(
        messages=[
            {"role": "user", "content": user_prompt}
        ],
        system_prompt=system_prompt,
        model=model,
        max_tokens=512,
        temperature=0
    ):
        content = chunk["choices"][0]["message"]["content"]
        response += content
        logger.success(content, flush=True)
    enhanced_query = response.strip()
    logger.debug(f"Enhanced query: {enhanced_query}")
    query_embedding = embed_func(enhanced_query)
    initial_results = vector_store.search(query_embedding, top_k=k*2)
    ranked_results = []
    for doc in initial_results:
        relevance_score = score_document_relevance(
            enhanced_query, doc["text"], mlx, model)
        ranked_results.append({
            "text": doc["text"],
            "metadata": doc["metadata"],
            "similarity": doc["similarity"],
            "relevance_score": relevance_score
        })
    ranked_results.sort(key=lambda x: x["relevance_score"], reverse=True)
    return ranked_results[:k]


def analytical_retrieval_strategy(query: str, vector_store: SimpleVectorStore, embed_func, mlx, k: int = 4, model=None) -> List[Dict[str, Any]]:
    """Analytical retrieval strategy with sub-queries."""
    logger.debug(f"Executing Analytical retrieval strategy for: '{query}'")
    system_prompt = "Generate a list of sub-questions for this analytical query, one per line."
    user_prompt = f"Query: {query}"
    response = ""
    for chunk in gen.stream_chat(
        messages=[
            {"role": "user", "content": user_prompt}
        ],
        system_prompt=system_prompt,
        model=model,
        max_tokens=512,
        temperature=0
    ):
        content = chunk["choices"][0]["message"]["content"]
        response += content
        logger.success(content, flush=True)
    sub_queries = response.strip().split(
        '\n')
    sub_queries = [q.strip() for q in sub_queries if q.strip()]
    logger.debug(f"Generated sub-queries: {sub_queries}")
    all_results = []
    for sub_query in sub_queries:
        sub_query_embedding = embed_func(sub_query)
        results = vector_store.search(sub_query_embedding, top_k=2)
        all_results.extend(results)
    unique_texts = set()
    diverse_results = []
    for result in all_results:
        if result["text"] not in unique_texts:
            unique_texts.add(result["text"])
            diverse_results.append(result)
    if len(diverse_results) < k:
        main_query_embedding = embed_func(query)
        main_results = vector_store.search(main_query_embedding, top_k=k)
        for result in main_results:
            if result["text"] not in unique_texts and len(diverse_results) < k:
                unique_texts.add(result["text"])
                diverse_results.append(result)
    return diverse_results[:k]


def opinion_retrieval_strategy(query: str, vector_store: SimpleVectorStore, embed_func, mlx, k: int = 4, model=None) -> List[Dict[str, Any]]:
    """Opinion retrieval strategy with viewpoint diversity."""
    logger.debug(f"Executing Opinion retrieval strategy for: '{query}'")
    system_prompt = "Identify different perspectives on this query, one per line."
    user_prompt = f"Query: {query}"
    response = ""
    for chunk in gen.stream_chat(
        messages=[
            {"role": "user", "content": user_prompt}
        ],
        system_prompt=system_prompt,
        model=model,
        max_tokens=512,
        temperature=0
    ):
        content = chunk["choices"][0]["message"]["content"]
        response += content
        logger.success(content, flush=True)
    viewpoints = response.strip().split(
        '\n')
    viewpoints = [v.strip() for v in viewpoints if v.strip()]
    logger.debug(f"Identified viewpoints: {viewpoints}")
    all_results = []
    for viewpoint in viewpoints:
        combined_query = f"{query} {viewpoint}"
        viewpoint_embedding = embed_func(combined_query)
        results = vector_store.search(viewpoint_embedding, top_k=2)
        for result in results:
            result["viewpoint"] = viewpoint
        all_results.extend(results)
    selected_results = []
    for viewpoint in viewpoints:
        viewpoint_docs = [r for r in all_results if r.get(
            "viewpoint") == viewpoint]
        if viewpoint_docs:
            selected_results.append(viewpoint_docs[0])
    remaining_slots = k - len(selected_results)
    if remaining_slots > 0:
        remaining_docs = [r for r in all_results if r not in selected_results]
        remaining_docs.sort(key=lambda x: x["similarity"], reverse=True)
        selected_results.extend(remaining_docs[:remaining_slots])
    return selected_results[:k]


def contextual_retrieval_strategy(query: str, vector_store: SimpleVectorStore, embed_func, mlx, k: int = 4, user_context: str = None, model=None) -> List[Dict[str, Any]]:
    """Contextual retrieval strategy with context enhancement."""
    logger.debug(f"Executing Contextual retrieval strategy for: '{query}'")
    if not user_context:
        system_prompt = "Infer the implied context in this query. Respond with only the inferred context."
        user_prompt = f"Query: {query}"
        response = ""
        for chunk in gen.stream_chat(
            messages=[
                {"role": "user", "content": user_prompt}
            ],
            system_prompt=system_prompt,
            model=model,
            max_tokens=512,
            temperature=0
        ):
            content = chunk["choices"][0]["message"]["content"]
            response += content
            logger.success(content, flush=True)
        user_context = response.strip()
        logger.debug(f"Inferred context: {user_context}")
    system_prompt = "Combine the query with the provided context to create a contextualized query."
    user_prompt = f"Query: {query}\nContext: {user_context}"
    response = ""
    for chunk in gen.stream_chat(
        messages=[
            {"role": "user", "content": user_prompt}
        ],
        system_prompt=system_prompt,
        model=model,
        max_tokens=512,
        temperature=0
    ):
        content = chunk["choices"][0]["message"]["content"]
        response += content
        logger.success(content, flush=True)
    contextualized_query = response.strip()
    logger.debug(f"Contextualized query: {contextualized_query}")
    query_embedding = embed_func(contextualized_query)
    initial_results = vector_store.search(query_embedding, top_k=k*2)
    ranked_results = []
    for doc in initial_results:
        context_relevance = score_document_context_relevance(
            query, user_context, doc["text"], mlx, model)
        ranked_results.append({
            "text": doc["text"],
            "metadata": doc["metadata"],
            "similarity": doc["similarity"],
            "context_relevance": context_relevance
        })
    ranked_results.sort(key=lambda x: x["context_relevance"], reverse=True)
    return ranked_results[:k]


def score_document_relevance(query: str, document: str, mlx, model=None) -> float:
    """Score document relevance to query."""
    doc_preview = document[:1500] + "..." if len(document) > 1500 else document
    system_prompt = "Score the relevance of the document to the query from 0 to 10, where 10 is highly relevant. Provide only the score."
    user_prompt = f"Query: {query}\nDocument: {doc_preview}"
    response = ""
    for chunk in gen.stream_chat(
        messages=[
            {"role": "user", "content": user_prompt}
        ],
        system_prompt=system_prompt,
        model=model,
        max_tokens=512,
        temperature=0
    ):
        content = chunk["choices"][0]["message"]["content"]
        response += content
        logger.success(content, flush=True)
    score_text = response.strip()
    score_match = re.search(r'\b(10|[0-9])\b', score_text)
    return float(score_match.group(1)) if score_match else 5.0


def score_document_context_relevance(query: str, context: str, document: str, mlx, model=None) -> float:
    """Score document relevance to query and context."""
    doc_preview = document[:1500] + "..." if len(document) > 1500 else document
    system_prompt = "Score the relevance of the document to the query and context from 0 to 10, where 10 is highly relevant. Provide only the score."
    user_prompt = f"Query: {query}\nContext: {context}\nDocument: {doc_preview}"
    response = ""
    for chunk in gen.stream_chat(
        messages=[
            {"role": "user", "content": user_prompt}
        ],
        system_prompt=system_prompt,
        model=model,
        max_tokens=512,
        temperature=0
    ):
        content = chunk["choices"][0]["message"]["content"]
        response += content
        logger.success(content, flush=True)
    score_text = response.strip()
    score_match = re.search(r'\b(10|[0-9])\b', score_text)
    return float(score_match.group(1)) if score_match else 5.0


def adaptive_retrieval(query: str, vector_store: SimpleVectorStore, embed_func, mlx, k: int = 4, user_context: str = None, model=None) -> List[Dict[str, Any]]:
    """Perform adaptive retrieval based on query type."""
    query_type = classify_query(query, mlx, model)
    logger.debug(f"Query classified as: {query_type}")
    if query_type == "Factual":
        results = factual_retrieval_strategy(
            query, vector_store, embed_func, mlx, k, model)
    elif query_type == "Analytical":
        results = analytical_retrieval_strategy(
            query, vector_store, embed_func, mlx, k, model)
    elif query_type == "Opinion":
        results = opinion_retrieval_strategy(
            query, vector_store, embed_func, mlx, k, model)
    elif query_type == "Contextual":
        results = contextual_retrieval_strategy(
            query, vector_store, embed_func, mlx, k, user_context, model)
    else:
        results = factual_retrieval_strategy(
            query, vector_store, embed_func, mlx, k, model)
    return results


def rag_with_adaptive_retrieval(chunks: List[Dict[str, Any]], query: str, embed_func, mlx, k: int = 4, user_context: str = None, model=None) -> Dict[str, Any]:
    """Run RAG with adaptive retrieval."""
    logger.debug("\n=== RAG WITH ADAPTIVE RETRIEVAL ===")
    logger.debug(f"Query: {query}")
    text_chunks, vector_store = process_document(chunks, embed_func)
    query_type = classify_query(query, mlx, model)
    logger.debug(f"Query classified as: {query_type}")
    retrieved_docs = adaptive_retrieval(
        query, vector_store, embed_func, mlx, k, user_context, model)
    system_prompt = "You are a helpful assistant. Answer the question based on the provided context. If you cannot answer from the context, acknowledge the limitations."
    response = generate_ai_response(
        query, system_prompt, retrieved_docs, mlx, logger, model=model)
    result = {
        "query": query,
        "query_type": query_type,
        "retrieved_documents": retrieved_docs,
        "response": response
    }
    logger.debug("\n=== RESPONSE ===")
    logger.debug(response)
    return result


def evaluate_adaptive_vs_standard(chunks: List[Dict[str, Any]], test_queries: List[str], embed_func, mlx, reference_answers: List[str] = None, model=None) -> Dict[str, Any]:
    """Evaluate adaptive vs standard retrieval."""
    logger.debug("=== EVALUATING ADAPTIVE VS. STANDARD RETRIEVAL ===")
    text_chunks, vector_store = process_document(chunks, embed_func)
    results = []
    for i, query in enumerate(test_queries):
        logger.debug(f"\n\nQuery {i+1}: {query}")
        logger.debug("\n--- Standard Retrieval ---")
        query_embedding = embed_func(query)
        standard_docs = vector_store.search(query_embedding, top_k=4)
        standard_response = generate_ai_response(
            query, "You are a helpful assistant. Answer the question based on the provided context.", standard_docs, mlx, logger, model=model)
        logger.debug("\n--- Adaptive Retrieval ---")
        query_type = classify_query(query, mlx, model)
        adaptive_docs = adaptive_retrieval(
            query, vector_store, embed_func, mlx, k=4, model=model)
        adaptive_response = generate_ai_response(
            query, "You are a helpful assistant. Answer the question based on the provided context.", adaptive_docs, mlx, logger, model=model)
        result = {
            "query": query,
            "query_type": query_type,
            "standard_retrieval": {
                "documents": standard_docs,
                "response": standard_response
            },
            "adaptive_retrieval": {
                "documents": adaptive_docs,
                "response": adaptive_response
            }
        }
        if reference_answers and i < len(reference_answers):
            result["reference_answer"] = reference_answers[i]
        results.append(result)
        logger.debug("\n--- Responses ---")
        logger.debug(f"Standard: {standard_response[:200]}...")
        logger.debug(f"Adaptive: {adaptive_response[:200]}...")
    comparison = compare_responses(
        results, mlx, model) if reference_answers else "No reference answers provided for evaluation"
    return {
        "results": results,
        "comparison": comparison
    }


def compare_responses(results: List[Dict[str, Any]], mlx, model=None) -> str:
    """Compare standard and adaptive responses."""
    comparison_text = ""
    system_prompt = "You are an objective evaluator. Compare the responses and provide a concise evaluation."
    for i, result in enumerate(results):
        if "reference_answer" not in result:
            continue
        comparison_text += f"\n\n**Query {i+1}: {result['query']}**\n"
        comparison_text += f"*Query Type: {result['query_type']}*\n\n"
        comparison_text += f"**Reference Answer:**\n{result['reference_answer']}\n\n"
        comparison_text += f"**Standard Retrieval Response:**\n{result['standard_retrieval']['response']}\n\n"
        comparison_text += f"**Adaptive Retrieval Response:**\n{result['adaptive_retrieval']['response']}\n\n"
        user_prompt = f"Reference Answer: {result['reference_answer']}\n\nStandard Response:\n{result['standard_retrieval']['response']}\n\nAdaptive Response:\n{result['adaptive_retrieval']['response']}"
        response = ""
        for chunk in gen.stream_chat(
            messages=[
                {"role": "user", "content": user_prompt}
            ],
            system_prompt=system_prompt,
            model=model,
            max_tokens=512,
            temperature=0
        ):
            content = chunk["choices"][0]["message"]["content"]
            response += content
            logger.success(content, flush=True)
        comparison_text += f"**Comparison Analysis:**\n{response}\n\n"
    return comparison_text


script_dir, generated_dir, log_file, logger = setup_config(__file__)
mlx, embed_func = initialize_mlx(logger)
formatted_texts, original_chunks = load_json_data(DOCS_PATH, logger)
logger.info("Loaded pre-chunked data from DOCS_PATH")
test_queries = [
    "What are the most popular isekai anime of 2025?",
    "Which 2025 isekai anime features a protagonist who is content with a low-profile life despite being overpowered?",
    "What is the release date and studio for Onmyo Kaiten Re:Birth Verse?",
    "Which 2025 isekai anime involves a yandere dark elf following the protagonist to Earth?",
    "What unique mechanic does Welcome to Japan, Ms. Elf! use for world-hopping?",
    "Which 2025 isekai anime has a protagonist reincarnated as a villainess in an otome game but struggles to act villainous?",
    "What is the premise of The Beginning After the End, and why has its anime adaptation been disappointing?",
    "Which 2025 isekai anime focuses on a protagonist using strategy game expertise to build an evil civilization?",
    "What makes Lord of Mysteries stand out among 2025 isekai anime?",
    "Which 2025 isekai anime sequel continues the story of a protagonist with an absurd skill for cooking?",
    "What is the unique hook of Otherworldly Munchkin: Let's Speedrun the Dungeon with Only 1 HP!?",
]
reference_answers = [
    "The most popular isekai anime of 2025, based on fan rankings and critical reception, include: \n1. **From Bureaucrat to Villainess: Dad's Been Reincarnated!** - Praised for its humorous take on the villainess subgenre, with a middle-aged dad reincarnated as an otome game villainess, becoming a fan favorite for its comedic charm (Ranker, 2025). \n2. **Lord of Mysteries** - A standout Donghua with a unique Victorian steampunk aesthetic and gorgeous animation, earning high praise for its intriguing narrative (CBR, 2025). \n3. **Apocalypse Bringer Mynoghra** - Noted for its strategy game-inspired premise and engaging world-building, appealing to fans of unique isekai settings (ScreenRant, 2025). \n4. **The Water Magician** - Popular for its cozy yet action-packed vibe and stunning visuals, offering a fresh take on the genre (ScreenRant, 2025). \n5. **The Beginning After the End** - Despite mixed reviews due to animation quality, its strong source material and fanbase make it highly anticipated (Ranker, 2025). \nThese titles have garnered significant attention for their innovative premises and engaging storytelling.",
    "A Gatherer's Adventure in Isekai features a protagonist who, despite being incredibly overpowered, prefers to keep a low profile, complete quests, and live humbly.",
    "Onmyo Kaiten Re:Birth Verse is set to release on July 2, 2025, and is produced by David Production.",
    "Yandere Dark Elf: She Chased Me All the Way From Another World! is a reverse isekai anime where a yandere dark elf follows the male protagonist to Earth, released on April 7, 2025.",
    "Welcome to Japan, Ms. Elf! uses a unique mechanic where Kazuhiro visits a fantasy world while dreaming and can bring someone back to Earth by holding them when he dies or sleeps in that realm.",
    "From Bureaucrat to Villainess: Dad's Been Reincarnated! features a middle-aged man reincarnated as an otome game's villainess who tries to follow the story but ends up acting like a supportive dad instead.",
    "The Beginning After the End follows a ruthless king reborn in a fantasy world with magic and a loving family, but the 2025 anime adaptation by Studio A-Cat has been disappointing due to lackluster animation and fast pacing.",
    "Apocalypse Bringer Mynoghra, released on July 6, 2025, features Takuto Ira, who uses his strategy game expertise to build the evil civilization Mynoghra from scratch.",
    "Lord of Mysteries, a Donghua released on June 28, 2025, stands out due to its Victorian-era steampunk and Gothic aesthetic, gorgeous animation, and intriguing storytelling.",
    "Campfire Cooking in Another World with My Absurd Skill Season 2, produced by MAPPA, continues the story of a protagonist with a unique cooking ability.",
    "Otherworldly Munchkin: Let's Speedrun the Dungeon with Only 1 HP! features a protagonist, a tabletop player, who uses a book granting dungeon and monster information to navigate an RPG world with only 1 HP, released in October 2025.",
]
evaluation_results = evaluate_adaptive_vs_standard(
    original_chunks, test_queries, embed_func, mlx, reference_answers
)
save_file(evaluation_results, f"{generated_dir}/evaluation_results.json")
logger.info(
    f"Saved evaluation results to {generated_dir}/evaluation_results.json")
logger.debug(evaluation_results["comparison"])
logger.info("\n\n[DONE]", bright=True)
