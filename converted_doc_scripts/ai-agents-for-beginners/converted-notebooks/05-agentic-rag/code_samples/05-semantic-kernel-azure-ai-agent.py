import asyncio
from jet.transformers.formatters import format_json
from azure.ai.projects.models import FileSearchTool, OllamaFile, VectorStore
from azure.identity.aio import DefaultAzureCredential
from jet.logger import CustomLogger
from semantic_kernel.agents import AzureAIAgent, AzureAIAgentThread
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# RAG Using Azure AI Agent Service & Semantic Kernel

This code snippet demonstrates how to create and manage an Azure AI agent for retrieval-augmented generation (RAG) using the `Azure AI Agent Service` and `Semantic Kernel`. The agent processes user queries based on the retrieved context and provides accurate responses accordingly.

## Initializing the Environment

SQLite Version Fix
If you encounter the error:
```
RuntimeError: Your system has an unsupported version of sqlite3. Chroma requires sqlite3 >= 3.35.0
```

Uncomment this code block at the start of your notebook:
"""
logger.info("# RAG Using Azure AI Agent Service & Semantic Kernel")



"""
### Importing Packages
The following code imports the necessary packages:
"""
logger.info("### Importing Packages")



"""
# Retrieval-Augmented Generation with Semantic Kernel & Azure AI Agent Service

This sample demonstrates how to use the **Azure AI Agent Service** to perform **Retrieval-Augmented Generation (RAG)** by combining a language model with domain-specific context from an uploaded document.

### How It Works

1. **Document Upload**: A markdown file (document.md) containing information (Contoso's travel insurance policy) is uploaded to the agent service.

2. **Vector Store Creation**: The document is indexed into a vector store to enable semantic search over its contents.

3. **Agent Configuration**: An agent is instantiated using the `gpt-4o` model with the following strict instructions:
   - Only answer questions based on retrieved content from the document.
   - Decline to answer if the question is out of scope.

4. **File Search Tool Integration**: The `FileSearchTool` is registered with the agent, enabling the model to search and retrieve relevant snippets from the indexed document during inference.

5. **User Interaction**: Users can ask questions. If relevant information is found in the document, the agent generates a grounded answer.  
   If not, the agent explicitly responds that the document does not contain sufficient information.

### Main Function

Make sure to first run `az login` using the Azure CLI so that the proper authentication context is provided while using the `DefaultAzureCredential`. The Azure AI Agent Service does not use API keys.
"""
logger.info("# Retrieval-Augmented Generation with Semantic Kernel & Azure AI Agent Service")

async def main():
    async def async_func_1():
        async with (
            DefaultAzureCredential() as creds,
            AzureAIAgent.create_client(credential=creds) as client,
        return result

    result = asyncio.run(async_func_1())
    logger.success(format_json(result))
    ):
        async def run_async_code_63d5bf5c():
            async def run_async_code_ca3ffc57():
                file: OllamaFile = await client.agents.upload_file_and_poll(file_path="document.md", purpose="assistants")
                return file: OllamaFile
            file: OllamaFile = asyncio.run(run_async_code_ca3ffc57())
            logger.success(format_json(file: OllamaFile))
            return file: OllamaFile
        file: OllamaFile = asyncio.run(run_async_code_63d5bf5c())
        logger.success(format_json(file: OllamaFile))
        async def async_func_6():
            vector_store: VectorStore = await client.agents.create_vector_store_and_poll(
                file_ids=[file.id], name="my_vectorstore"
            )
            return vector_store: VectorStore
        vector_store: VectorStore = asyncio.run(async_func_6())
        logger.success(format_json(vector_store: VectorStore))

        AGENT_NAME = "RAGAgent"
        AGENT_INSTRUCTIONS = """
        You are an AI assistant designed to answer user questions using only the information retrieved from the provided document(s).

        - If a user's question cannot be answered using the retrieved context, **you must clearly respond**:
        "I'm sorry, but the uploaded document does not contain the necessary information to answer that question."
        - Do not answer from general knowledge or reasoning. Do not make assumptions or generate hypothetical explanations.
        - Do not provide definitions, tutorials, or commentary that is not explicitly grounded in the content of the uploaded file(s).
        - If a user asks a question like "What is a Neural Network?", and this is not discussed in the uploaded document, respond as instructed above.
        - For questions that do have relevant content in the document (e.g., Contoso's travel insurance coverage), respond accurately, and cite the document explicitly.

        You must behave as if you have no external knowledge beyond what is retrieved from the uploaded document.
        """


        file_search = FileSearchTool(vector_store_ids=[vector_store.id])

        async def async_func_27():
            agent_definition = await client.agents.create_agent(
                model="llama3.1", request_timeout=300.0, context_window=4096,  # This model should match your Azure Ollama deployment.
                name=AGENT_NAME,
                instructions=AGENT_INSTRUCTIONS,
                tools=file_search.definitions,
                tool_resources=file_search.resources,
            )
            return agent_definition
        agent_definition = asyncio.run(async_func_27())
        logger.success(format_json(agent_definition))

        agent = AzureAIAgent(
            client=client,
            definition=agent_definition,
        )

        thread: AzureAIAgentThread | None = None

        user_inputs = [
            "Can you explain Contoso's travel insurance coverage?",  # Relevant context.
            "What is a Neural Network?"  # No relevant context from the document. Will not contain a source annotation.
        ]

        try:
            for user_input in user_inputs:
                logger.debug(f"# User: '{user_input}'")
                async for response in agent.invoke(messages=user_input, thread=thread):
                    logger.debug(f"# {response.name}: {response}")
                    thread = response.thread
        finally:
            async def run_async_code_8c2324e7():
                await thread.delete() if thread else None
                return 
             = asyncio.run(run_async_code_8c2324e7())
            logger.success(format_json())
            async def run_async_code_a3ff5ec9():
                await client.agents.delete_vector_store(vector_store.id)
                return 
             = asyncio.run(run_async_code_a3ff5ec9())
            logger.success(format_json())
            async def run_async_code_c13ee8e6():
                await client.agents.delete_file(file.id)
                return 
             = asyncio.run(run_async_code_c13ee8e6())
            logger.success(format_json())
            async def run_async_code_ffa0a0cc():
                await client.agents.delete_agent(agent.id)
                return 
             = asyncio.run(run_async_code_ffa0a0cc())
            logger.success(format_json())
            logger.debug("\nCleaned up agent, thread, file, and vector store.")

async def run_async_code_ba09313d():
    await main()
    return 
 = asyncio.run(run_async_code_ba09313d())
logger.success(format_json())

"""
You should see output similar to:

```
# User: 'Can you explain Contoso's travel insurance coverage?'
# Agent: Contoso's travel insurance coverage includes protection for medical emergencies, trip cancellations, and lost baggage【4:0†document.md】.
# User: 'What is a Neural Network?'
# Agent: I'm sorry, but the uploaded document does not contain the necessary information to answer that question.

Cleaned up agent, thread, file, and vector store.
```
"""
logger.info("# User: 'Can you explain Contoso's travel insurance coverage?'")

logger.info("\n\n[DONE]", bright=True)