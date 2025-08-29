async def main():
    from jet.transformers.formatters import format_json
    from jet.logger import CustomLogger
    from llama_cloud_services import LlamaParse
    from tqdm.asyncio import tqdm
    from typing import List, Tuple
    from zeroentropy import AsyncZeroEntropy, ConflictError
    import asyncio
    import io
    import os
    import requests
    import shutil
    import zipfile
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")
    
    """
    <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/cookbooks/rerank_llamaparsed
    _pdfs.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    
    # Reranking top pages from PDF using LlamaParse and ZeroEntropy
    
    In this guide, we'll build a simple workflow to parse PDF documents into text using LlamaParse and then query and rerank the textual data. 
    
    ---
    
    ### Pre-requisites
    - Python 3.8+
    - `zeroentropy` client
    - `llama_cloud_services` client
    - A ZeroEntropy API key ([Get yours here](https://dashboard.zeroentropy.dev))
    - A LlamaParse API key ([Get yours here](https://docs.cloud.llamaindex.ai/api_key))
    
    ### What You'll Learn
    - How to use LlamaParse to accurately convert PDF documents into markdown
    - How to use ZeroEntropy to semantically index and query the parsed documents
    - How to rerank your results using [ZeroEntropy's reranker zerank-1](https://www.zeroentropy.dev/blog/announcing-zeroentropys-first-reranker) to boost accuracy
    
    ### Setting up your ZeroEntropy Client and LlamaParse Client
    
    First, install dependencies:
    """
    logger.info("# Reranking top pages from PDF using LlamaParse and ZeroEntropy")
    
    # !pip install zeroentropy python-dotenv llama_cloud_services requests
    
    """
    Now load your API keys and initialize the clients
    """
    logger.info("Now load your API keys and initialize the clients")
    
    ZEROENTROPY_API_KEY = "your_api_key_here"
    LLAMAPARSE_API_KEY = "your_api_key_here"
    
    
    zclient = AsyncZeroEntropy(api_key=ZEROENTROPY_API_KEY)
    
    llamaParser = LlamaParse(
        api_key=LLAMAPARSE_API_KEY,
        num_workers=1,  # if multiple files passed, split in `num_workers` API calls
        result_type="text",
        verbose=True,
        language="en",  # optionally define a language, default=en
    )
    
    """
    ### Adding a collection to the ZeroEntropy client
    """
    logger.info("### Adding a collection to the ZeroEntropy client")
    
    collection_name = "my_collection"
    await zclient.collections.add(collection_name=collection_name)
    
    """
    Now define a function to download and extract PDF files from Dropbox directly to memory:
    """
    logger.info("Now define a function to download and extract PDF files from Dropbox directly to memory:")
    
    
    
    def download_and_extract_dropbox_zip_to_memory(
        url: str,
    ) -> List[Tuple[str, bytes]]:
        """Download and extract a zip file from Dropbox URL directly to memory.
    
        Returns:
            List of tuples containing (filename, file_content_bytes)
        """
        try:
            logger.debug(f"Downloading zip file from: {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
    
            zip_content = io.BytesIO()
            for chunk in response.iter_content(chunk_size=8192):
                zip_content.write(chunk)
            zip_content.seek(0)
    
            files_in_memory = []
            with zipfile.ZipFile(zip_content, "r") as zip_ref:
                for file_info in zip_ref.infolist():
                    if (
                        not file_info.is_dir()
                        and file_info.filename.lower().endswith(".pdf")
                    ):
                        file_content = zip_ref.read(file_info.filename)
                        files_in_memory.append((file_info.filename, file_content))
                        logger.debug(
                            f"Loaded {file_info.filename} ({len(file_content)} bytes)"
                        )
    
            logger.debug(
                f"Successfully loaded {len(files_in_memory)} PDF files into memory"
            )
            return files_in_memory
    
        except Exception as e:
            logger.debug(f"Error downloading/extracting zip file: {e}")
            raise
    
    
    dropbox_url = "https://www.dropbox.com/scl/fi/oi6kf91gz8h76d2wt57mb/example_docs.zip?rlkey=mf21tvyb65tyrjkr1t2szt226&dl=1"
    files_in_memory = download_and_extract_dropbox_zip_to_memory(dropbox_url)
    
    """
    ### Parsing PDFs using LlamaParse
    
    Let's download the PDF files from Dropbox and parse them directly in memory using LlamaParse:
    """
    logger.info("### Parsing PDFs using LlamaParse")
    
    file_objects = []
    file_names = []
    
    for filename, file_content in files_in_memory:
        file_obj = io.BytesIO(file_content)
        file_obj.name = filename  # Set the name attribute for LlamaParse
        file_objects.append(file_obj)
        file_names.append(filename)
    
    logger.debug(f"Parsing {len(file_objects)} PDF files...")
    
    text_data = await asyncio.gather(
            *[
                llamaParser.aparse(file_obj, extra_info={"file_name": name})
                for file_obj, name in zip(file_objects, file_names)
            ]
        )
    logger.success(format_json(text_data))
    logger.debug(f"Successfully parsed {len(text_data)} documents")
    
    """
    ## Organizing your documents
    
    Once parsed, we form a list of documents with a list of the pages within them.
    """
    logger.info("## Organizing your documents")
    
    docs = []
    
    for dindex, doc in enumerate(text_data):
        pages = []
        for index, page in enumerate(doc.pages):
            pages.append(page.text)
        docs.append(pages)
    
    logger.debug(f"Organized {len(docs)} documents with pages")
    if docs:
        logger.debug(f"First document has {len(docs[0])} pages")
    
    """
    ## Querying with ZeroEntropy
    We'll now define functions to upload the documents as text pages asynchroniously.
    """
    logger.info("## Querying with ZeroEntropy")
    
    
    sem = asyncio.Semaphore(16)
    
    
    async def add_document_with_pages(
        collection_name: str, filename: str, pages: list, doc_index: int
    ):
        """Add a single document with multiple pages to the collection."""
        async with sem:  # Limit concurrent operations
                for retry in range(3):  # Retry logic
                    try:
                        response = await zclient.documents.add(
                            collection_name=collection_name,
                            path=filename,  # Use the actual filename as path
                            content={
                                "type": "text-pages",
                                "pages": pages,  # Send list of strings directly
                            },
                        )
                        return response
                    except ConflictError:
                        logger.debug(
                            f"Document '{filename}' already exists in collection '{collection_name}'"
                        )
                        break
                    except Exception as e:
                        if retry == 2:  # Last retry
                            logger.debug(f"Failed to add document '{filename}': {e}")
                            return None
                        await asyncio.sleep(0.1 * (retry + 1))  # Exponential backoff
            
            
        logger.success(format_json(result))
    async def upload_documents_async(
        docs: list, file_names: list, collection_name: str
    ):
        """
        Upload documents asynchronously to ZeroEntropy collection.
    
        Args:
            docs: 2D array where docs[i] contains the list of pages (strings) for document i
            file_names: Array where file_names[i] contains the path for document i
            collection_name: Name of the collection to add documents to
        """
    
        if len(docs) != len(file_names):
            raise ValueError("docs and file_names must have the same length")
    
        logger.debug(f"Starting upload of {len(docs)} documents...")
    
        tasks = [
            add_document_with_pages(collection_name, file_names[i], docs[i], i)
            for i in range(len(docs))
        ]
    
        results = await tqdm.gather(*tasks, desc="Uploading Documents")
        logger.success(format_json(results))
    
        successful = sum(1 for result in results if result is not None)
        logger.debug(f"Successfully uploaded {successful}/{len(docs)} documents")
    
        return results
    
    """
    ### Querying documents with ZeroEntropy
    First we will upload documents
    """
    logger.info("### Querying documents with ZeroEntropy")
    
    await upload_documents_async(docs, file_names, collection_name)
    
    """
    Query for the top 5 pages
    """
    logger.info("Query for the top 5 pages")
    
    response = await zclient.queries.top_pages(
            collection_name=collection_name,
            query="What are the top 100 stocks in the S&P 500?",
            k=5,
        )
    logger.success(format_json(response))
    
    """
    Now let's define a function to rerank the pages in the response:
    """
    logger.info("Now let's define a function to rerank the pages in the response:")
    
    async def rerank_top_pages_with_metadata(
        query: str, top_pages_response, collection_name: str
    ):
        """
        Rerank the results from a top_pages query and return re-ordered list with metadata.
    
        Args:
            query: The query string to use for reranking
            top_pages_response: The response object from zclient.queries.top_pages()
            collection_name: Name of the collection to fetch page content from
    
        Returns:
            List of dicts with 'path', 'page_index', and 'rerank_score' in reranked order
        """
    
        documents = []
        metadata = []
    
        for result in top_pages_response.results:
            page_info = await zclient.documents.get_page_info(
                    collection_name=collection_name,
                    path=result.path,
                    page_index=result.page_index,
                    include_content=True,
                )
            logger.success(format_json(page_info))
    
            page_content = page_info.page.content
            if page_content and page_content.strip():
                documents.append(page_content.strip())
                metadata.append(
                    {
                        "path": result.path,
                        "page_index": result.page_index,
                        "original_score": result.score,
                    }
                )
            else:
                documents.append("No content available")
                metadata.append(
                    {
                        "path": result.path,
                        "page_index": result.page_index,
                        "original_score": result.score,
                    }
                )
    
        if not documents:
            raise ValueError("No documents found to rerank")
    
        rerank_response = await zclient.models.rerank(
                model="zerank-1", query=query, documents=documents
            )
        logger.success(format_json(rerank_response))
    
        reranked_results = []
        for rerank_result in rerank_response.results:
            original_metadata = metadata[rerank_result.index]
            reranked_results.append(
                {
                    "path": original_metadata["path"],
                    "page_index": original_metadata["page_index"],
                    "rerank_score": rerank_result.relevance_score,
                }
            )
    
        return reranked_results
    
    """
    Run the function and see the results!
    """
    logger.info("Run the function and see the results!")
    
    reranked_results = await rerank_top_pages_with_metadata(
            query="What are the top 100 stocks in the S&P 500?",
            top_pages_response=response,
            collection_name=collection_name,
        )
    logger.success(format_json(reranked_results))
    
    logger.debug("Reranked Results with Metadata:")
    for i, result in enumerate(reranked_results, 1):
        logger.debug(
            f"Rank {i}: {result['path']} (Page {result['page_index']}) - Score: {result['rerank_score']:.4f}"
        )
    
    """
    ### ✅ That's It!
    
    You've now built a working semantic search engine that processes PDF files entirely in memory using ZeroEntropy and LlamaParse — no local file storage required!
    """
    logger.info("### ✅ That's It!")
    
    logger.info("\n\n[DONE]", bright=True)

if __name__ == '__main__':
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(main())
        else:
            loop.run_until_complete(main())
    except RuntimeError:
        asyncio.run(main())