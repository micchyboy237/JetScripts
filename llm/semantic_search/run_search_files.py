import shutil
from llama_index.core import Document, SimpleDirectoryReader, PromptTemplate
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES

from convert_docs_to_scripts import scrape_code
from script_utils import display_source_nodes
from jet.llm.token import token_counter
from jet.llm.query import setup_index, query_llm
from jet.logger import logger
from jet.llm.ollama import initialize_ollama_settings, small_embed_model
initialize_ollama_settings(settings={"embedding_model": small_embed_model})


def get_token_counts(texts, model):
    token_counts = token_counter(
        texts, model, prevent_total=True)
    max_count = max(token_counts)
    min_count = min(token_counts)
    total_count = sum(token_counts)
    return {
        "max": max_count,
        "min": min_count,
        "total": total_count,
    }


if __name__ == "__main__":
    model = "llama3.1"
    max_tokens = 0.5
    score_threshold = 0.2

    code_types = ["python", "text"]
    # sample_query = "Write real world usage example for tree summarizer with complete code and a main function. Respond only with the generated code as a single large code block."
    # sample_query = "Write usage examples for for sql_agent, ToolMetadata and OpenAIAgent."
    sample_query = "Write real world usage example for tree summarizer."
    # sample_query = "Create endpoints for the main features."

    results_dir = "generated/results"

    system = "You are an AI assistant that follows instructions. You can refactor code, understand and write code in any programming language, and extract code from unstructured web content. You are skilled in fixing bugs and syntax errors while ensuring the code is clean, optimized, readable, and modular. You explain code clearly and systematically using markdown, providing real-world usage examples to enhance understanding and usability."

    prompt_template = PromptTemplate(
        """\
    Context information are below.
    ---------------------
    {context_str}
    ---------------------
    Given the context information and not prior knowledge, respond to the question.
    Question: {query_str}
    Response: \
    """
    )

    data_dir = "/Users/jethroestrada/Desktop/External_Projects/AI/repo-libs/llama_index/docs/docs/understanding/putting_it_all_together"
    data_extensions = [".ipynb", ".md", ".mdx", ".rst"]
    rag_dir = "generated/rag"
    rag_extensions = [".py"]

    shutil.rmtree(rag_dir, ignore_errors=True)

    include_files = []
    exclude_files = []

    # rag_files = search_files(rag_dir, extensions,
    #                          include_files, exclude_files)
    # logger.info(f"Found {len(rag_files)} files with extensions {extensions}")

    results = scrape_code(
        data_dir,
        data_extensions,
        include_files=include_files,
        exclude_files=exclude_files,
        with_markdown=True,
        with_ollama=True,
        output_base_dir=rag_dir,
        types=code_types,
    )

    shutil.rmtree(results_dir, ignore_errors=True)

    documents = []
    for result in results:
        # code = "\n\n".join([
        #     block['code'] for idx, block in enumerate(result["blocks"])
        #     if block["type"] != 'text'
        # ]).strip()
        # description = next(
        #     (block['code']
        #      for block in result["blocks"] if block["type"] == 'text'),
        #     ""
        # ).strip()

        # # data_file = result['data_file']
        # file_rel_path = result['code_file'].replace(rag_dir, "").lstrip("/")
        # code = f"File: {file_rel_path}\nDescription: {
        #     description}\nCode:\n{code}"
        # doc = Document(text=code, metadata={
        #     "file_name": file_rel_path,
        # })
        # documents.append(doc)

        filtered_blocks = [
            block for idx, block in enumerate(result["blocks"])
            if block["type"] != 'text'
        ]
        # data_file = result['data_file']
        file_rel_path = result['code_file'].replace(rag_dir, "").lstrip("/")
        for idx, block in enumerate(filtered_blocks):
            file = file_rel_path
            part = idx + 1
            code = block["code"].strip()
            code = f"File: {file}\nPart: {part}\n{code}"
            doc = Document(text=code, metadata={
                "file_name": file,
                "part": part
            })
            documents.append(doc)

    # documents = SimpleDirectoryReader(
    #     rag_dir, required_exts=rag_extensions, recursive=True).load_data()

    chunk_size = 512
    chunk_overlap = 64
    top_k = 50

    query_nodes = setup_index(
        documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # logger.newline()
    # logger.info("RECIPROCAL_RANK: query...")
    # response = query_nodes(sample_query, FUSION_MODES.RECIPROCAL_RANK)

    # logger.newline()
    # logger.info("DIST_BASED_SCORE: query...")
    # response = query_nodes(sample_query, FUSION_MODES.DIST_BASED_SCORE)

    logger.newline()
    logger.info("RELATIVE_SCORE: sample query...")
    result = query_nodes(
        sample_query, FUSION_MODES.RELATIVE_SCORE, threshold=score_threshold, top_k=top_k)
    logger.info(f"RETRIEVED NODES ({len(result["nodes"])})")
    display_source_nodes(sample_query, result["nodes"])

    result_texts = [node.text for node in result["nodes"]]
    token_count = get_token_counts(result_texts, model)

    logger.info("Token counts")
    logger.log("Max:", token_count['max'], colors=["DEBUG", "SUCCESS"])
    logger.log("Min:", token_count['min'], colors=["DEBUG", "SUCCESS"])
    logger.log("Total:", token_count['total'], colors=["DEBUG", "SUCCESS"])

    instructions = "Generate complete working code and a main function. Respond only with the generated code as a single large code block."
    # contexts = [f"File: {node.metadata['file']}\nPart: {
    #     node.metadata['part']}\n{node.text}" for node in result["nodes"]]
    contexts = [
        f"File: {node.metadata['file_name']}\n{node.text}"
        for node in result["nodes"]
    ]
    response = query_llm(
        sample_query + "\n" + instructions,
        contexts,
        model=model,
        system=system,
        max_tokens=max_tokens,
    )

    # Run app
    while True:
        # Continuously ask user for queries
        try:
            query = input("Enter your query (type 'exit' to quit): ").strip()
            if query.lower() == "exit":
                print("Exiting query loop.")
                break

            result = query_nodes(
                query, FUSION_MODES.RELATIVE_SCORE, threshold=score_threshold, top_k=top_k)
            logger.info(f"RETRIEVED NODES ({len(result["nodes"])})")
            display_source_nodes(query, result["nodes"])

            result_texts = [node.text for node in result["nodes"]]
            token_count = get_token_counts(result_texts, model)

            logger.info("Token counts")
            logger.log("Max:", token_count['max'], colors=["DEBUG", "SUCCESS"])
            logger.log("Min:", token_count['min'], colors=["DEBUG", "SUCCESS"])
            logger.log("Total:", token_count['total'], colors=[
                       "DEBUG", "SUCCESS"])

            # contexts = [f"File: {node.metadata['file']}\nPart: {
            #     node.metadata['part']}\n{node.text}" for node in result["nodes"]]
            contexts = [
                f"File: {node.metadata['file_name']}\n{node.text}"
                for node in result["nodes"]
            ]
            response = query_llm(
                query + "\n" + instructions,
                contexts,
                model=model,
                max_tokens=max_tokens,
            )

        except KeyboardInterrupt:
            print("\nExiting query loop.")
            break
        except Exception as e:
            logger.error(f"Error while processing query: {e}")

    logger.info("\n\n[DONE]", bright=True)
