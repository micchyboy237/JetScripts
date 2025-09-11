from jet.transformers.formatters import format_json
from dotenv import load_dotenv
from jet.logger import logger
from langchain_cloudflare.embeddings import (
from langchain_cloudflare.vectorstores import (
from langchain_community.document_loaders import WikipediaLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import asyncio
import json
import os
import shutil
import uuid
import warnings

async def main():
    CloudflareWorkersAIEmbeddings,
    )
    CloudflareVectorize,
    )
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger.basicConfig(filename=log_file)
    logger.info(f"Logs: {log_file}")
    
    PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
    os.makedirs(PERSIST_DIR, exist_ok=True)
    
    """
    ---
    sidebar_label: CloudflareVectorize
    ---
    
    # CloudflareVectorizeVectorStore
    
    This notebook covers how to get started with the CloudflareVectorize vector store.
    
    ## Setup
    
    This Python package is a wrapper around Cloudflare's REST API.  To interact with the API, you need to provide an API token with the appropriate privileges.
    
    You can create and manage API tokens here:
    
    https://dash.cloudflare.com/YOUR-ACCT-NUMBER/api-tokens
    
    ### Credentials
    
    CloudflareVectorize depends on WorkersAI (if you want to use it for Embeddings), and D1 (if you are using it to store and retrieve raw values).
    
    While you can create a single `api_token` with Edit privileges to all needed resources (WorkersAI, Vectorize & D1), you may want to follow the principle of "least privilege access" and create separate API tokens for each service
    
    **Note:** These service-specific tokens (if provided) will take preference over a global token.  You could provide these instead of a global token.
    
    You can also leave these parameters as environmental variables.
    """
    logger.info("# CloudflareVectorizeVectorStore")
    
    
    
    load_dotenv(".env")
    
    cf_acct_id = os.getenv("CF_ACCOUNT_ID")
    
    api_token = os.getenv("CF_API_TOKEN")
    
    cf_vectorize_token = os.getenv("CF_VECTORIZE_API_TOKEN")
    cf_d1_token = os.getenv("CF_D1_API_TOKEN")
    
    """
    ## Initialization
    """
    logger.info("## Initialization")
    
    
    
    warnings.filterwarnings("ignore")
    
    vectorize_index_name = f"test-langchain-{uuid.uuid4().hex}"
    
    """
    ### Embeddings
    
    For storage of embeddings, semantic search and retrieval, you must embed your raw values as embeddings.  Specify an embedding model, one available on WorkersAI
    
    [https://developers.cloudflare.com/workers-ai/models/](https://developers.cloudflare.com/workers-ai/models/)
    """
    logger.info("### Embeddings")
    
    MODEL_WORKERSAI = "@cf/baai/bge-large-en-v1.5"
    
    cf_ai_token = os.getenv(
        "CF_AI_API_TOKEN"
    )  # needed if you want to use workersAI for embeddings
    
    embedder = CloudflareWorkersAIEmbeddings(
        account_id=cf_acct_id, api_token=cf_ai_token, model_name=MODEL_WORKERSAI
    )
    
    """
    ### Raw Values with D1
    
    Vectorize only stores embeddings, metadata and namespaces. If you want to store and retrieve raw values, you must leverage Cloudflare's SQL Database D1.
    
    You can create a database here and retrieve its id:
    
    [https://dash.cloudflare.com/YOUR-ACCT-NUMBER/workers/d1
    """
    logger.info("### Raw Values with D1")
    
    d1_database_id = os.getenv("CF_D1_DATABASE_ID")
    
    """
    ### CloudflareVectorize Class
    
    Now we can create the CloudflareVectorize instance.  Here we passed:
    
    * The `embedding` instance from earlier
    * The account ID
    * A global API token for all services (WorkersAI, Vectorize, D1)
    * Individual API tokens for each service
    """
    logger.info("### CloudflareVectorize Class")
    
    cfVect = CloudflareVectorize(
        embedding=embedder,
        account_id=cf_acct_id,
        d1_api_token=cf_d1_token,  # (Optional if using global token)
        vectorize_api_token=cf_vectorize_token,  # (Optional if using global token)
        d1_database_id=d1_database_id,  # (Optional if not using D1)
    )
    
    """
    ### Cleanup
    Before we get started, let's delete any `test-langchain*` indexes we have for this walkthrough
    """
    logger.info("### Cleanup")
    
    arr_indexes = cfVect.list_indexes()
    arr_indexes = [x for x in arr_indexes if "test-langchain" in x.get("name")]
    arr_async_requests = [
        cfVect.adelete_index(index_name=x.get("name")) for x in arr_indexes
    ]
    await asyncio.gather(*arr_async_requests);
    
    """
    ### Gotchyas
    
    D1 Database ID provided but no "global" `api_token` and no `d1_api_token`
    """
    logger.info("### Gotchyas")
    
    try:
        cfVect = CloudflareVectorize(
            embedding=embedder,
            account_id=cf_acct_id,
            ai_api_token=cf_ai_token,  # (Optional if using global token)
            vectorize_api_token=cf_vectorize_token,  # (Optional if using global token)
            d1_database_id=d1_database_id,  # (Optional if not using D1)
        )
    except Exception as e:
        logger.debug(str(e))
    
    """
    ## Manage Vector Store
    
    ### Creating an Index
    
    Let's start off this example by creating and index (and first deleting if it exists).  If the index doesn't exist we will get a an error from Cloudflare telling us so.
    """
    logger.info("## Manage Vector Store")
    
    # %%capture
    
    try:
        cfVect.delete_index(index_name=vectorize_index_name, wait=True)
    except Exception as e:
        logger.debug(e)
    
    """
    
    """
    
    r = cfVect.create_index(
        index_name=vectorize_index_name, description="A Test Vectorize Index", wait=True
    )
    logger.debug(r)
    
    """
    ### Listing Indexes
    
    Now, we can list our indexes on our account
    """
    logger.info("### Listing Indexes")
    
    indexes = cfVect.list_indexes()
    indexes = [x for x in indexes if "test-langchain" in x.get("name")]
    logger.debug(indexes)
    
    """
    ### Get Index Info
    We can also get certain indexes and retrieve more granular information about an index.
    
    This call returns a `processedUpToMutation` which can be used to track the status of operations such as creating indexes, adding or deleting records.
    """
    logger.info("### Get Index Info")
    
    r = cfVect.get_index_info(index_name=vectorize_index_name)
    logger.debug(r)
    
    """
    ### Adding Metadata Indexes
    
    It is common to assist retrieval by supplying metadata filters in quereies.  In Vectorize, this is accomplished by first creating a "metadata index" on your Vectorize Index.  We will do so for our example by creating one on the `section` field in our documents.
    
    **Reference:** [https://developers.cloudflare.com/vectorize/reference/metadata-filtering/](https://developers.cloudflare.com/vectorize/reference/metadata-filtering/)
    """
    logger.info("### Adding Metadata Indexes")
    
    r = cfVect.create_metadata_index(
        property_name="section",
        index_type="string",
        index_name=vectorize_index_name,
        wait=True,
    )
    logger.debug(r)
    
    """
    ### Listing Metadata Indexes
    """
    logger.info("### Listing Metadata Indexes")
    
    r = cfVect.list_metadata_indexes(index_name=vectorize_index_name)
    logger.debug(r)
    
    """
    ### Adding Documents
    For this example, we will use LangChain's Wikipedia loader to pull an article about Cloudflare.  We will store this in Vectorize and query its contents later.
    """
    logger.info("### Adding Documents")
    
    docs = WikipediaLoader(query="Cloudflare", load_max_docs=2).load()
    
    """
    We will then create some simple chunks with metadata based on the chunk sections.
    """
    logger.info("We will then create some simple chunks with metadata based on the chunk sections.")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.create_documents([docs[0].page_content])
    
    running_section = ""
    for idx, text in enumerate(texts):
        if text.page_content.startswith("="):
            running_section = text.page_content
            running_section = running_section.replace("=", "").strip()
        else:
            if running_section == "":
                text.metadata = {"section": "Introduction"}
            else:
                text.metadata = {"section": running_section}
    
    logger.debug(len(texts))
    logger.debug(texts[0], "\n\n", texts[-1])
    
    """
    Now we will add documents to our Vectorize Index.
    
    **Note:**
    Adding embeddings to Vectorize happens `asyncronously`, meaning there will be a small delay between adding the embeddings and being able to query them.  By default `add_documents` has a `wait=True` parameter which waits for this operation to complete before returning a response.  If you do not want the program to wait for embeddings availability, you can set this to `wait=False`.
    """
    logger.info("Now we will add documents to our Vectorize Index.")
    
    r = cfVect.add_documents(index_name=vectorize_index_name, documents=texts, wait=True)
    
    logger.debug(json.dumps(r)[:300])
    
    """
    ## Query vector store
    
    We will do some searches on our embeddings.  We can specify our search `query` and the top number of results we want with `k`.
    """
    logger.info("## Query vector store")
    
    query_documents = cfVect.similarity_search(
        index_name=vectorize_index_name, query="Workers AI", k=100, return_metadata="none"
    )
    
    logger.debug(f"{len(query_documents)} results:\n{query_documents[:3]}")
    
    """
    ### Output
    
    If you want to return metadata you can pass `return_metadata="all" | 'indexed'`.  The default is `all`.
    
    If you want to return the embeddings values, you can pass `return_values=True`.  The default is `False`.
    Embeddings will be returned in the `metadata` field under the special `_values` field.
    
    **Note:** `return_metadata="none"` and `return_values=True` will return only ther `_values` field in `metadata`.
    
    **Note:**
    If you return metadata or values, the results will be limited to the top 20.
    
    [https://developers.cloudflare.com/vectorize/platform/limits/](https://developers.cloudflare.com/vectorize/platform/limits/)
    """
    logger.info("### Output")
    
    query_documents = cfVect.similarity_search(
        index_name=vectorize_index_name,
        query="Workers AI",
        return_values=True,
        return_metadata="all",
        k=100,
    )
    logger.debug(f"{len(query_documents)} results:\n{str(query_documents[0])[:500]}")
    
    """
    If you'd like the similarity `scores` to be returned, you can use `similarity_search_with_score`
    """
    logger.info("If you'd like the similarity `scores` to be returned, you can use `similarity_search_with_score`")
    
    query_documents = cfVect.similarity_search_with_score(
        index_name=vectorize_index_name,
        query="Workers AI",
        k=100,
        return_metadata="all",
    )
    logger.debug(f"{len(query_documents)} results:\n{str(query_documents[0])[:500]}")
    
    """
    ### Including D1 for "Raw Values"
    All of the `add` and `search` methods on CloudflareVectorize support a `include_d1` parameter (default=True).
    
    This is to configure whether you want to store/retrieve raw values.
    
    If you do not want to use D1 for this, you can set this to `include=False`.  This will return documents with an empty `page_content` field.
    
    **Note:** Your D1 table name MUST MATCH your vectorize index name!  If you run 'create_index' and include_d1=True or  cfVect(d1_database=...,) this D1 table will be created along with your Vectorize Index.
    """
    logger.info("### Including D1 for "Raw Values"")
    
    query_documents = cfVect.similarity_search_with_score(
        index_name=vectorize_index_name,
        query="california",
        k=100,
        return_metadata="all",
        include_d1=False,
    )
    logger.debug(f"{len(query_documents)} results:\n{str(query_documents[0])[:500]}")
    
    """
    ### Query by turning into retriever
    
    You can also transform the vector store into a retriever for easier usage in your chains.
    """
    logger.info("### Query by turning into retriever")
    
    retriever = cfVect.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1, "index_name": vectorize_index_name},
    )
    r = retriever.get_relevant_documents("california")
    
    """
    ### Searching with Metadata Filtering
    
    As mentioned before, Vectorize supports filtered search via filtered on indexes metadata fields.  Here is an example where we search for `Introduction` values within the indexed `section` metadata field.
    
    More info on searching on Metadata fields is here: [https://developers.cloudflare.com/vectorize/reference/metadata-filtering/](https://developers.cloudflare.com/vectorize/reference/metadata-filtering/)
    """
    logger.info("### Searching with Metadata Filtering")
    
    query_documents = cfVect.similarity_search_with_score(
        index_name=vectorize_index_name,
        query="California",
        k=100,
        md_filter={"section": "Introduction"},
        return_metadata="all",
    )
    logger.debug(f"{len(query_documents)} results:\n - {str(query_documents[:3])}")
    
    """
    You can do more sophisticated filtering as well
    
    https://developers.cloudflare.com/vectorize/reference/metadata-filtering/#valid-filter-examples
    """
    logger.info("You can do more sophisticated filtering as well")
    
    query_documents = cfVect.similarity_search_with_score(
        index_name=vectorize_index_name,
        query="California",
        k=100,
        md_filter={"section": {"$ne": "Introduction"}},
        return_metadata="all",
    )
    logger.debug(f"{len(query_documents)} results:\n - {str(query_documents[:3])}")
    
    query_documents = cfVect.similarity_search_with_score(
        index_name=vectorize_index_name,
        query="DNS",
        k=100,
        md_filter={"section": {"$in": ["Products", "History"]}},
        return_metadata="all",
    )
    logger.debug(f"{len(query_documents)} results:\n - {str(query_documents)}")
    
    """
    ### Search by Namespace
    We can also search for vectors by `namespace`.  We just need to add it to the `namespaces` array when adding it to our vector database.
    
    https://developers.cloudflare.com/vectorize/reference/metadata-filtering/#namespace-versus-metadata-filtering
    """
    logger.info("### Search by Namespace")
    
    namespace_name = f"test-namespace-{uuid.uuid4().hex[:8]}"
    
    new_documents = [
        Document(
            page_content="This is a new namespace specific document!",
            metadata={"section": "Namespace Test1"},
        ),
        Document(
            page_content="This is another namespace specific document!",
            metadata={"section": "Namespace Test2"},
        ),
    ]
    
    r = cfVect.add_documents(
        index_name=vectorize_index_name,
        documents=new_documents,
        namespaces=[namespace_name] * len(new_documents),
        wait=True,
    )
    
    query_documents = cfVect.similarity_search(
        index_name=vectorize_index_name,
        query="California",
        namespace=namespace_name,
    )
    
    logger.debug(f"{len(query_documents)} results:\n - {str(query_documents)}")
    
    """
    ### Search by IDs
    We can also retrieve specific records for specific IDs.  To do so, we need to set the vectorize index name on the `index_name` Vectorize state param.
    
    This will return both `_namespace` and `_values` as well as other `metadata`.
    """
    logger.info("### Search by IDs")
    
    sample_ids = [x.id for x in query_documents]
    
    cfVect.index_name = vectorize_index_name
    
    query_documents = cfVect.get_by_ids(
        sample_ids,
    )
    logger.debug(str(query_documents[:3])[:500])
    
    """
    The namespace will be included in the `_namespace` field in `metadata` along with your other metadata (if you requested it in `return_metadata`).
    
    **Note:** You cannot set the `_namespace` or `_values` fields in `metadata` as they are reserved.  They will be stripped out during the insert process.
    
    ### Upserts
    
    Vectorize supports Upserts which you can perform by setting `upsert=True`.
    """
    logger.info("### Upserts")
    
    query_documents[0].page_content = "Updated: " + query_documents[0].page_content
    logger.debug(query_documents[0].page_content)
    
    new_document_id = "12345678910"
    new_document = Document(
        id=new_document_id,
        page_content="This is a new document!",
        metadata={"section": "Introduction"},
    )
    
    r = cfVect.add_documents(
        index_name=vectorize_index_name,
        documents=[new_document, query_documents[0]],
        upsert=True,
        wait=True,
    )
    
    query_documents_updated = cfVect.get_by_ids([new_document_id, query_documents[0].id])
    
    logger.debug(str(query_documents_updated[0])[:500])
    logger.debug(query_documents_updated[0].page_content)
    logger.debug(query_documents_updated[1].page_content)
    
    """
    ### Deleting Records
    We can delete records by their ids as well
    """
    logger.info("### Deleting Records")
    
    r = cfVect.delete(index_name=vectorize_index_name, ids=sample_ids, wait=True)
    logger.debug(r)
    
    """
    And to confirm deletion
    """
    logger.info("And to confirm deletion")
    
    query_documents = cfVect.get_by_ids(sample_ids)
    assert len(query_documents) == 0
    
    """
    ### Creating from Documents
    LangChain stipulates that all vectorstores must have a `from_documents` method to instantiate a new Vectorstore from documents.  This is a more streamlined method than the individual `create, add` steps shown above.
    
    You can do that as shown here:
    """
    logger.info("### Creating from Documents")
    
    vectorize_index_name = "test-langchain-from-docs"
    
    cfVect = CloudflareVectorize.from_documents(
        account_id=cf_acct_id,
        index_name=vectorize_index_name,
        documents=texts,
        embedding=embedder,
        d1_database_id=d1_database_id,
        d1_api_token=cf_d1_token,
        vectorize_api_token=cf_vectorize_token,
        wait=True,
    )
    
    query_documents = cfVect.similarity_search(
        index_name=vectorize_index_name,
        query="Edge Computing",
    )
    
    logger.debug(f"{len(query_documents)} results:\n{str(query_documents[0])[:300]}")
    
    """
    ## Async Examples
    This section will show some Async examples
    
    ### Creating Indexes
    """
    logger.info("## Async Examples")
    
    vectorize_index_name1 = f"test-langchain-{uuid.uuid4().hex}"
    vectorize_index_name2 = f"test-langchain-{uuid.uuid4().hex}"
    
    async_requests = [
        cfVect.acreate_index(index_name=vectorize_index_name1),
        cfVect.acreate_index(index_name=vectorize_index_name2),
    ]
    
    res = await asyncio.gather(*async_requests);
    logger.success(format_json(res))
    
    """
    ### Creating Metadata Indexes
    """
    logger.info("### Creating Metadata Indexes")
    
    async_requests = [
        cfVect.acreate_metadata_index(
            property_name="section",
            index_type="string",
            index_name=vectorize_index_name1,
            wait=True,
        ),
        cfVect.acreate_metadata_index(
            property_name="section",
            index_type="string",
            index_name=vectorize_index_name2,
            wait=True,
        ),
    ]
    
    await asyncio.gather(*async_requests);
    
    """
    ### Adding Documents
    """
    logger.info("### Adding Documents")
    
    async_requests = [
        cfVect.aadd_documents(index_name=vectorize_index_name1, documents=texts, wait=True),
        cfVect.aadd_documents(index_name=vectorize_index_name2, documents=texts, wait=True),
    ]
    
    await asyncio.gather(*async_requests);
    
    """
    ### Querying/Search
    """
    logger.info("### Querying/Search")
    
    async_requests = [
        cfVect.asimilarity_search(index_name=vectorize_index_name1, query="Workers AI"),
        cfVect.asimilarity_search(index_name=vectorize_index_name2, query="Edge Computing"),
    ]
    
    async_results = await asyncio.gather(*async_requests);
    logger.success(format_json(async_results))
    
    logger.debug(f"{len(async_results[0])} results:\n{str(async_results[0][0])[:300]}")
    logger.debug(f"{len(async_results[1])} results:\n{str(async_results[1][0])[:300]}")
    
    """
    ### Returning Metadata/Values
    """
    logger.info("### Returning Metadata/Values")
    
    async_requests = [
        cfVect.asimilarity_search(
            index_name=vectorize_index_name1,
            query="California",
            return_values=True,
            return_metadata="all",
        ),
        cfVect.asimilarity_search(
            index_name=vectorize_index_name2,
            query="California",
            return_values=True,
            return_metadata="all",
        ),
    ]
    
    async_results = await asyncio.gather(*async_requests);
    logger.success(format_json(async_results))
    
    logger.debug(f"{len(async_results[0])} results:\n{str(async_results[0][0])[:300]}")
    logger.debug(f"{len(async_results[1])} results:\n{str(async_results[1][0])[:300]}")
    
    """
    ### Searching with Metadata Filtering
    """
    logger.info("### Searching with Metadata Filtering")
    
    async_requests = [
        cfVect.asimilarity_search(
            index_name=vectorize_index_name1,
            query="Cloudflare services",
            k=2,
            md_filter={"section": "Products"},
            return_metadata="all",
        ),
        cfVect.asimilarity_search(
            index_name=vectorize_index_name2,
            query="Cloudflare services",
            k=2,
            md_filter={"section": "Products"},
            return_metadata="all",
        ),
    ]
    
    async_results = await asyncio.gather(*async_requests);
    logger.success(format_json(async_results))
    
    logger.debug(f"{len(async_results[0])} results:\n{str(async_results[0][-1])[:300]}")
    logger.debug(f"{len(async_results[1])} results:\n{str(async_results[1][0])[:300]}")
    
    """
    ## Cleanup
    Let's finish by deleting all of the indexes we created in this notebook.
    """
    logger.info("## Cleanup")
    
    arr_indexes = cfVect.list_indexes()
    arr_indexes = [x for x in arr_indexes if "test-langchain" in x.get("name")]
    
    arr_async_requests = [
        cfVect.adelete_index(index_name=x.get("name")) for x in arr_indexes
    ]
    await asyncio.gather(*arr_async_requests);
    
    """
    ## API Reference
    
    https://developers.cloudflare.com/api/resources/vectorize/
    
    https://developers.cloudflare.com/vectorize/
    """
    logger.info("## API Reference")
    
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