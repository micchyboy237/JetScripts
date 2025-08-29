async def main():
    from jet.models.config import MODELS_CACHE_DIR
    from jet.transformers.formatters import format_json
    from jet.logger import CustomLogger
    from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.core.tools import QueryEngineTool
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.bedrock_converse import BedrockConverse
    import os
    import shutil
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")
    
    """
    <a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/agent/bedrock_converse_agent.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    
    # Function Calling AWS Bedrock Converse Agent
    
    This notebook shows you how to use our AWS Bedrock Converse agent, powered by function calling capabilities.
    
    ## Initial Setup
    
    Let's start by importing some simple building blocks.  
    
    The main thing we need is:
    1. AWS credentials with access to Bedrock and the Claude Haiku LLM
    2. a place to keep conversation history 
    3. a definition for tools that our agent can use.
    
    If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
    """
    logger.info("# Function Calling AWS Bedrock Converse Agent")
    
    # %pip install llama-index
    # %pip install llama-index-llms-bedrock-converse
    # %pip install llama-index-embeddings-huggingface
    
    """
    Let's define some very simple calculator tools for our agent.
    """
    logger.info("Let's define some very simple calculator tools for our agent.")
    
    def multiply(a: int, b: int) -> int:
        """Multiple two integers and returns the result integer"""
        return a * b
    
    
    def add(a: int, b: int) -> int:
        """Add two integers and returns the result integer"""
        return a + b
    
    """
    Make sure to set your AWS credentials, either the `profile_name` or the keys below.
    """
    logger.info("Make sure to set your AWS credentials, either the `profile_name` or the keys below.")
    
    
    llm = BedrockConverse(
        model="anthropic.claude-3-haiku-20240307-v1:0",
        aws_access_key_id="AWS Access Key ID to use",
        aws_secret_access_key="AWS Secret Access Key to use",
        aws_session_token="AWS Session Token to use",
        region_name="AWS Region to use, eg. us-east-1",
    )
    
    """
    ## Initialize AWS Bedrock Converse Agent
    
    Here we initialize a simple AWS Bedrock Converse agent with calculator functions.
    """
    logger.info("## Initialize AWS Bedrock Converse Agent")
    
    
    agent = FunctionAgent(
        tools=[multiply, add],
        llm=llm,
    )
    
    """
    ### Chat
    """
    logger.info("### Chat")
    
    response = await agent.run("What is (121 + 2) * 5?")
    logger.success(format_json(response))
    logger.success(format_json(response))
    logger.debug(str(response))
    
    logger.debug(response.tool_calls)
    
    """
    ## AWS Bedrock Converse Agent over RAG Pipeline
    
    Build an AWS Bedrock Converse agent over a simple 10K document. We use both HuggingFace embeddings and `BAAI/bge-small-en-v1.5` to construct the RAG pipeline, and pass it to the AWS Bedrock Converse agent as a tool.
    """
    logger.info("## AWS Bedrock Converse Agent over RAG Pipeline")
    
    # !mkdir -p 'data/10k/'
    # !curl -o 'data/10k/uber_2021.pdf' 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/10k/uber_2021.pdf'
    
    
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR)
    query_llm = BedrockConverse(
        model="anthropic.claude-3-haiku-20240307-v1:0",
        aws_access_key_id="AWS Access Key ID to use",
        aws_secret_access_key="AWS Secret Access Key to use",
        aws_session_token="AWS Session Token to use",
        region_name="AWS Region to use, eg. us-east-1",
    )
    
    uber_docs = SimpleDirectoryReader(
        input_files=["./data/10k/uber_2021.pdf"]
    ).load_data()
    
    uber_index = VectorStoreIndex.from_documents(
        uber_docs, embed_model=embed_model
    )
    uber_engine = uber_index.as_query_engine(similarity_top_k=3, llm=query_llm)
    query_engine_tool = QueryEngineTool.from_defaults(
        query_engine=uber_engine,
        name="uber_10k",
        description=(
            "Provides information about Uber financials for year 2021. "
            "Use a detailed plain text question as input to the tool."
        ),
    )
    
    
    agent = FunctionAgent(
        tools=[query_engine_tool],
        llm=llm,
    )
    
    response = await agent.run(
            "Tell me both the risk factors and tailwinds for Uber? Do two parallel tool calls."
        )
    logger.success(format_json(response))
    
    logger.debug(str(response))
    
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