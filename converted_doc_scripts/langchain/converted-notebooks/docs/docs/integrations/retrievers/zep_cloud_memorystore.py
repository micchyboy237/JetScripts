from jet.logger import logger
from langchain_community.memory.zep_cloud_memory import ZepCloudMemory
from langchain_community.retrievers import ZepCloudRetriever
from langchain_core.messages import AIMessage, HumanMessage
from uuid import uuid4
import os
import shutil
import time

async def main():
    
    
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
    # Zep Cloud
    ## Retriever Example for [Zep Cloud](https://docs.getzep.com/)
    
    > Recall, understand, and extract data from chat histories. Power personalized AI experiences.
    
    > [Zep](https://www.getzep.com) is a long-term memory service for AI Assistant apps.
    > With Zep, you can provide AI assistants with the ability to recall past conversations, no matter how distant,
    > while also reducing hallucinations, latency, and cost.
    
    > See [Zep Cloud Installation Guide](https://help.getzep.com/sdks) and more [Zep Cloud Langchain Examples](https://github.com/getzep/zep-python/tree/main/examples)
    
    ## Retriever Example
    
    This notebook demonstrates how to search historical chat message histories using the [Zep Long-term Memory Store](https://www.getzep.com/).
    
    We'll demonstrate:
    
    1. Adding conversation history to the Zep memory store.
    2. Vector search over the conversation history: 
        1. With a similarity search over chat messages
        2. Using maximal marginal relevance re-ranking of a chat message search
        3. Filtering a search using metadata filters
        4. A similarity search over summaries of the chat messages
        5. Using maximal marginal relevance re-ranking of a summary search
    """
    logger.info("# Zep Cloud")
    
    # import getpass
    
    
    # zep_api_key = getpass.getpass()
    
    """
    ### Initialize the Zep Chat Message History Class and add a chat message history to the memory store
    
    **NOTE:** Unlike other Retrievers, the content returned by the Zep Retriever is session/user specific. A `session_id` is required when instantiating the Retriever.
    """
    logger.info("### Initialize the Zep Chat Message History Class and add a chat message history to the memory store")
    
    session_id = str(uuid4())  # This is a unique identifier for the user/session
    
    zep_memory = ZepCloudMemory(session_id=session_id, api_key=zep_api_key)
    
    test_history = [
        {"role": "human", "role_type": "user", "content": "Who was Octavia Butler?"},
        {
            "role": "ai",
            "role_type": "assistant",
            "content": (
                "Octavia Estelle Butler (June 22, 1947 – February 24, 2006) was an American"
                " science fiction author."
            ),
        },
        {
            "role": "human",
            "role_type": "user",
            "content": "Which books of hers were made into movies?",
        },
        {
            "role": "ai",
            "role_type": "assistant",
            "content": (
                "The most well-known adaptation of Octavia Butler's work is the FX series"
                " Kindred, based on her novel of the same name."
            ),
        },
        {"role": "human", "role_type": "user", "content": "Who were her contemporaries?"},
        {
            "role": "ai",
            "role_type": "assistant",
            "content": (
                "Octavia Butler's contemporaries included Ursula K. Le Guin, Samuel R."
                " Delany, and Joanna Russ."
            ),
        },
        {"role": "human", "role_type": "user", "content": "What awards did she win?"},
        {
            "role": "ai",
            "role_type": "assistant",
            "content": (
                "Octavia Butler won the Hugo Award, the Nebula Award, and the MacArthur"
                " Fellowship."
            ),
        },
        {
            "role": "human",
            "role_type": "user",
            "content": "Which other women sci-fi writers might I want to read?",
        },
        {
            "role": "ai",
            "role_type": "assistant",
            "content": "You might want to read Ursula K. Le Guin or Joanna Russ.",
        },
        {
            "role": "human",
            "role_type": "user",
            "content": (
                "Write a short synopsis of Butler's book, Parable of the Sower. What is it"
                " about?"
            ),
        },
        {
            "role": "ai",
            "role_type": "assistant",
            "content": (
                "Parable of the Sower is a science fiction novel by Octavia Butler,"
                " published in 1993. It follows the story of Lauren Olamina, a young woman"
                " living in a dystopian future where society has collapsed due to"
                " environmental disasters, poverty, and violence."
            ),
        },
        {
            "role": "human",
            "role_type": "user",
            "content": "What is the setting of the book?",
        },
        {
            "role": "ai",
            "role_type": "assistant",
            "content": (
                "The book is set in a dystopian future in the 2020s, where society has"
                " collapsed due to climate change and economic crises."
            ),
        },
        {"role": "human", "role_type": "user", "content": "Who is the protagonist?"},
        {
            "role": "ai",
            "role_type": "assistant",
            "content": (
                "The protagonist of the book is Lauren Olamina, a young woman who possesses"
                " 'hyperempathy', the ability to feel pain and other sensations she"
                " witnesses."
            ),
        },
        {
            "role": "human",
            "role_type": "user",
            "content": "What is the main theme of the book?",
        },
        {
            "role": "ai",
            "role_type": "assistant",
            "content": (
                "The main theme of the book is survival in the face of drastic societal"
                " change and collapse. It also explores themes of adaptability, community,"
                " and the human capacity for change."
            ),
        },
        {
            "role": "human",
            "role_type": "user",
            "content": "What is the 'Parable of the Sower'?",
        },
        {
            "role": "ai",
            "role_type": "assistant",
            "content": (
                "The 'Parable of the Sower' is a biblical parable that Butler uses as a"
                " metaphor in the book. In the parable, a sower scatters seeds, some of"
                " which fall on fertile ground and grow, while others fall on rocky ground"
                " or among thorns and fail to grow. The parable is used to illustrate the"
                " importance of receptivity and preparedness in the face of change."
            ),
        },
        {
            "role": "human",
            "role_type": "user",
            "content": "What is Butler's writing style like?",
        },
        {
            "role": "ai",
            "role_type": "assistant",
            "content": (
                "Butler's writing style is known for its clarity, directness, and"
                " psychological insight. Her narratives often involve complex, diverse"
                " characters and explore themes of race, gender, and power."
            ),
        },
        {
            "role": "human",
            "role_type": "user",
            "content": "What other books has she written?",
        },
        {
            "role": "ai",
            "content": (
                "In addition to 'Parable of the Sower', Butler has written several other"
                " notable works, including 'Kindred', 'Dawn', and 'Parable of the Talents'."
            ),
        },
    ]
    
    for msg in test_history:
        zep_memory.chat_memory.add_message(
            HumanMessage(content=msg["content"])
            if msg["role"] == "human"
            else AIMessage(content=msg["content"])
        )
    
    time.sleep(
        10
    )  # Wait for the messages to be embedded and summarized, this happens asynchronously.
    
    """
    ### Use the Zep Retriever to vector search over the Zep memory
    
    Zep provides native vector search over historical conversation memory. Embedding happens automatically.
    
    NOTE: Embedding of messages occurs asynchronously, so the first query may not return results. Subsequent queries will return results as the embeddings are generated.
    """
    logger.info("### Use the Zep Retriever to vector search over the Zep memory")
    
    zep_retriever = ZepCloudRetriever(
        api_key=zep_api_key,
        session_id=session_id,  # Ensure that you provide the session_id when instantiating the Retriever
        top_k=5,
    )
    
    await zep_retriever.ainvoke("Who wrote Parable of the Sower?")
    
    """
    We can also use the Zep sync API to retrieve results:
    """
    logger.info("We can also use the Zep sync API to retrieve results:")
    
    zep_retriever.invoke("Who wrote Parable of the Sower?")
    
    """
    ### Reranking using MMR (Maximal Marginal Relevance)
    
    Zep has native, SIMD-accelerated support for reranking results using MMR. This is useful for removing redundancy in results.
    """
    logger.info("### Reranking using MMR (Maximal Marginal Relevance)")
    
    zep_retriever = ZepCloudRetriever(
        api_key=zep_api_key,
        session_id=session_id,  # Ensure that you provide the session_id when instantiating the Retriever
        top_k=5,
        search_type="mmr",
        mmr_lambda=0.5,
    )
    
    await zep_retriever.ainvoke("Who wrote Parable of the Sower?")
    
    """
    ### Using metadata filters to refine search results
    
    Zep supports filtering results by metadata. This is useful for filtering results by entity type, or other metadata.
    
    More information here: https://help.getzep.com/document-collections#searching-a-collection-with-hybrid-vector-search
    """
    logger.info("### Using metadata filters to refine search results")
    
    filter = {"where": {"jsonpath": '$[*] ? (@.baz == "qux")'}}
    
    await zep_retriever.ainvoke(
        "Who wrote Parable of the Sower?", config={"metadata": filter}
    )
    
    """
    ### Searching over Summaries with MMR Reranking
    
    Zep automatically generates summaries of chat messages. These summaries can be searched over using the Zep Retriever. Since a summary is a distillation of a conversation, they're more likely to match your search query and offer rich, succinct context to the LLM.
    
    Successive summaries may include similar content, with Zep's similarity search returning the highest matching results but with little diversity.
    MMR re-ranks the results to ensure that the summaries you populate into your prompt are both relevant and each offers additional information to the LLM.
    """
    logger.info("### Searching over Summaries with MMR Reranking")
    
    zep_retriever = ZepCloudRetriever(
        api_key=zep_api_key,
        session_id=session_id,  # Ensure that you provide the session_id when instantiating the Retriever
        top_k=3,
        search_scope="summary",
        search_type="mmr",
        mmr_lambda=0.5,
    )
    
    await zep_retriever.ainvoke("Who wrote Parable of the Sower?")
    
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