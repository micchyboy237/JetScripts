async def main():
    from jet.transformers.formatters import format_json
    from PIL import Image
    from dotenv import load_dotenv
    from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
    from jet.logger import CustomLogger
    from llama_index.core.agent import FunctionAgent
    from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent
    from llama_index.core.agent.workflow import FunctionAgent
    from llama_index.core.llms import ChatMessage, ImageBlock, TextBlock
    from llama_index.core.tools import FunctionTool
    from llama_index.core.tools import FunctionTool  # Import FunctionTool
    from maxim import Config, Maxim
    from maxim.logger import LoggerConfig
    from maxim.logger.llamaindex import instrument_llamaindex
    import asyncio
    import base64
    import io
    import os
    import requests
    import shutil
    
    
    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")
    
    """
    ## Setup and Installation
    
    <a href="https://colab.research.google.com/drive/1vysj4eGqYt4sBdMp1BDI5Kxp4jtQbBxZ?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
    
    # Cookbook LlamaIndex Integration by Maxim AI (Instrumentation Module)
    
    This is a simple cookbook that demonstrates how to use the [LlamaIndex Maxim integration](https://www.getmaxim.ai/docs/sdk/python/integrations/llamaindex/llamaindex) using the [instrumentation module](https://docs.llamaindex.ai/en/stable/module_guides/observability/instrumentation/) by LlamaIndex (available in llama-index v0.10.20 and later).
    
    <img 
      src="https://cdn.getmaxim.ai/public/images/llamaindex.gif" 
      alt="LlamaIndex demo gif"
      style="max-width: 100%; height: auto; border-radius: 10px;"
    />
    """
    logger.info("## Setup and Installation")
    
    
    
    
    load_dotenv()
    
    MAXIM_API_KEY = os.getenv("MAXIM_API_KEY")
    MAXIM_LOG_REPO_ID = os.getenv("MAXIM_LOG_REPO_ID")
    # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    if not MAXIM_API_KEY:
        raise ValueError("MAXIM_API_KEY environment variable is required")
    if not MAXIM_LOG_REPO_ID:
        raise ValueError("MAXIM_LOG_REPO_ID environment variable is required")
    # if not OPENAI_API_KEY:
    #     raise ValueError("OPENAI_API_KEY environment variable is required")
    
    logger.debug("✅ Environment variables loaded successfully")
    logger.debug(
        f"MAXIM_API_KEY: {'*' * (len(MAXIM_API_KEY) - 4) + MAXIM_API_KEY[-4:] if MAXIM_API_KEY else 'Not set'}"
    )
    logger.debug(f"MAXIM_LOG_REPO_ID: {MAXIM_LOG_REPO_ID}")
    logger.debug(
    #     f"OPENAI_API_KEY: {'*' * (len(OPENAI_API_KEY) - 4) + OPENAI_API_KEY[-4:] if OPENAI_API_KEY else 'Not set'}"
    )
    
    """
    ## Maxim Configuration
    """
    logger.info("## Maxim Configuration")
    
    
    maxim = Maxim(Config(api_key=os.getenv("MAXIM_API_KEY")))
    logger = maxim.logger(LoggerConfig(id=os.getenv("MAXIM_LOG_REPO_ID")))
    
    instrument_llamaindex(logger)
    
    logger.debug("✅ Maxim instrumentation enabled for LlamaIndex")
    
    """
    ## Simple FunctionAgent with Observability
    """
    logger.info("## Simple FunctionAgent with Observability")
    
    
    
    def add_numbers(a: float, b: float) -> float:
        """Add two numbers together."""
        return a + b
    
    
    def multiply_numbers(a: float, b: float) -> float:
        """Multiply two numbers together."""
        return a * b
    
    
    def divide_numbers(a: float, b: float) -> float:
        """Divide first number by second number."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    
    
    add_tool = FunctionTool.from_defaults(fn=add_numbers)
    multiply_tool = FunctionTool.from_defaults(fn=multiply_numbers)
    divide_tool = FunctionTool.from_defaults(fn=divide_numbers)
    
    llm = OllamaFunctionCallingAdapter(model="llama3.2", temperature=0)
    
    agent = FunctionAgent(
        tools=[add_tool, multiply_tool, divide_tool],
        llm=llm,
        verbose=True,
        system_prompt="""You are a helpful calculator assistant.
        Use the provided tools to perform mathematical calculations.
        Always explain your reasoning step by step.""",
    )
    
    
    
    async def test_function_agent():
        logger.debug("🔍 Testing FunctionAgent with Maxim observability...")
    
        query = "What is (15 + 25) multiplied by 2, then divided by 8?"
    
        logger.debug(f"\n📝 Query: {query}")
    
        response = await agent.run(query)
        logger.success(format_json(response))
        logger.success(format_json(response))
    
        logger.debug(f"\n🤖 Response: {response}")
        logger.debug("\n✅ Check your Maxim dashboard for detailed trace information!")
    
    
    await test_function_agent()
    
    """
    ## Multi Modal Requests
    """
    logger.info("## Multi Modal Requests")
    
    
    
    def describe_image_content(description: str) -> str:
        """Analyze and describe what's in an image based on the model's vision."""
        return f"Image analysis complete: {description}"
    
    
    def add(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b
    
    
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers together."""
        return a * b
    
    
    multimodal_llm = OllamaFunctionCallingAdapter(model="llama3.2")  # Vision-capable model
    
    multimodal_agent = FunctionAgent(
        tools=[add, multiply, describe_image_content],
        llm=multimodal_llm,
        system_prompt="You are a helpful assistant that can analyze images and perform calculations.",
    )
    
    
    async def test_multimodal_agent():
        logger.debug("🔍 Testing Multi-Modal Agent with Maxim observability...")
    
        try:
    
    
            msg = ChatMessage(
                role="user",
                blocks=[
                    TextBlock(
                        text="What do you see in this image? If there are numbers, perform calculations."
                    ),
                    ImageBlock(
                        url="https://www.shutterstock.com/image-photo/simple-mathematical-equation-260nw-350386472.jpg"
                    ),  # Replace with actual image path
                ],
            )
            response = await multimodal_agent.run(msg)
            logger.success(format_json(response))
            logger.success(format_json(response))
    
        except Exception as e:
            logger.debug(
                f"Note: Multi-modal features require actual image files. Error: {e}"
            )
            logger.debug(
                "The agent structure is set up correctly for when you have images to process!"
            )
    
        logger.debug("\n✅ Check Maxim dashboard for multi-modal agent traces!")
    
    
    await test_multimodal_agent()
    
    """
    ## Multiple Agents
    """
    logger.info("## Multiple Agents")
    
    
    
    def research_topic(topic: str) -> str:
        """Research a given topic and return key findings."""
        research_data = {
            "climate change": "Climate change refers to long-term shifts in global temperatures and weather patterns, primarily caused by human activities since the 1800s.",
            "renewable energy": "Renewable energy comes from sources that are naturally replenishing like solar, wind, hydro, and geothermal power.",
            "artificial intelligence": "AI involves creating computer systems that can perform tasks typically requiring human intelligence.",
            "sustainability": "Sustainability involves meeting present needs without compromising the ability of future generations to meet their needs.",
        }
    
        topic_lower = topic.lower()
        for key, info in research_data.items():
            if key in topic_lower:
                return f"Research findings on {topic}: {info} Additional context includes recent developments and policy implications."
    
        return f"Research completed on {topic}. This is an emerging area requiring further investigation and analysis."
    
    
    def analyze_data(research_data: str) -> str:
        """Analyze research data and provide insights."""
        if "climate change" in research_data.lower():
            return "Analysis indicates climate change requires immediate action through carbon reduction, renewable energy adoption, and international cooperation."
        elif "renewable energy" in research_data.lower():
            return "Analysis shows renewable energy is becoming cost-competitive with fossil fuels and offers long-term economic and environmental benefits."
        elif "artificial intelligence" in research_data.lower():
            return "Analysis reveals AI has transformative potential across industries but requires careful consideration of ethical implications and regulation."
        else:
            return "Analysis suggests this topic has significant implications requiring strategic planning and stakeholder engagement."
    
    
    def write_report(analysis: str, topic: str) -> str:
        """Write a comprehensive report based on analysis."""
        return f"""
    ═══════════════════════════════════════
    COMPREHENSIVE RESEARCH REPORT: {topic.upper()}
    ═══════════════════════════════════════
    
    EXECUTIVE SUMMARY:
    {analysis}
    
    KEY FINDINGS:
    - Evidence-based analysis indicates significant implications
    - Multiple stakeholder perspectives must be considered
    - Implementation requires coordinated approach
    - Long-term monitoring and evaluation necessary
    
    RECOMMENDATIONS:
    1. Develop comprehensive strategy framework
    2. Engage key stakeholders early in process
    3. Establish clear metrics and milestones
    4. Create feedback mechanisms for continuous improvement
    5. Allocate appropriate resources and timeline
    
    NEXT STEPS:
    - Schedule stakeholder consultations
    - Develop detailed implementation plan
    - Establish monitoring and evaluation framework
    - Begin pilot program if applicable
    
    This report provides a foundation for informed decision-decision making and strategic planning.
    """
    
    
    llm = OllamaFunctionCallingAdapter(model="llama3.2", temperature=0)
    
    research_agent = FunctionAgent(
        name="research_agent",
        description="This agent researches a given topic and returns key findings.",
        tools=[FunctionTool.from_defaults(fn=research_topic)],
        llm=llm,
        system_prompt="You are a research specialist. Use the research tool to gather comprehensive information on requested topics.",
    )
    
    analysis_agent = FunctionAgent(
        name="analysis_agent",
        description="This agent analyzes research data and provides actionable insights.",
        tools=[FunctionTool.from_defaults(fn=analyze_data)],
        llm=llm,
        system_prompt="You are a data analyst. Analyze research findings and provide actionable insights.",
    )
    
    report_agent = FunctionAgent(
        name="report_agent",
        description="This agent creates comprehensive, well-structured reports based on analysis.",
        tools=[FunctionTool.from_defaults(fn=write_report)],
        llm=llm,
        system_prompt="You are a report writer. Create comprehensive, well-structured reports based on analysis.",
    )
    
    multi_agent_workflow = AgentWorkflow(
        agents=[research_agent, analysis_agent, report_agent],
        root_agent="research_agent",
    )
    
    
    async def test_agent_workflow():
        logger.debug("🔍 Testing AgentWorkflow with Maxim observability...")
    
        query = """I need a comprehensive report on renewable energy.
        Please research the current state of renewable energy,
        analyze the key findings, and create a structured report
        with recommendations for implementation."""
    
        logger.debug(f"\n📝 Query: {query}")
        logger.debug("🔄 This will coordinate multiple agents...")
    
        response = await multi_agent_workflow.run(query)
        logger.success(format_json(response))
        logger.success(format_json(response))
    
        logger.debug(f"\n🤖 Multi-Agent Response:\n{response}")
        logger.debug(
            "\n✅ Check Maxim dashboard for comprehensive multi-agent workflow traces!"
        )
    
    
    await test_agent_workflow()
    
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