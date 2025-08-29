async def main():
    from jet.models.config import MODELS_CACHE_DIR
    from jet.transformers.formatters import format_json
    from googleapiclient import discovery
    from jet.llm.ollama.adapters.ollama_llama_index_llm_adapter import OllamaFunctionCallingAdapter
    from jet.logger import CustomLogger
    from llama_index.agent.introspective import (
        ToolInteractiveReflectionAgentWorker,
    )
    from llama_index.agent.introspective import IntrospectiveAgentWorker
    from llama_index.agent.openai import OllamaFunctionCallingAdapterAgent
    from llama_index.core import SimpleDirectoryReader
    from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
    from llama_index.core import VectorStoreIndex
    from llama_index.core.agent import FunctionAgent
    from llama_index.core.bridge.pydantic import BaseModel, Field
    from llama_index.core.ingestion import IngestionPipeline
    from llama_index.core.llms import MessageRole, ChatMessage
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.prompts import ChatPromptTemplate
    from llama_index.core.prompts import PromptTemplate
    from llama_index.core.tools import FunctionTool
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.llms.anthropic import Anthropic
    from llama_index.llms.cohere import Cohere
    from llama_index.llms.mistralai import MistralAI
    from llama_index.readers.wikipedia import WikipediaReader
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    from numpy import random
    from typing import Dict, Optional, Tuple
    from typing import Literal, List, Optional
    import asyncio
    import os
    import qdrant_client
    import shutil

    OUTPUT_DIR = os.path.join(
        os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    log_file = os.path.join(OUTPUT_DIR, "main.log")
    logger = CustomLogger(log_file, overwrite=True)
    logger.info(f"Logs: {log_file}")

    """
    **NOTE:** This notebook was written in 2024, and is not guaranteed to work with the latest version of llama-index. It is presented here for reference only.
    
    ![Slide One](https://d3ddy8balm3goa.cloudfront.net/rbc-llm-workshop/1.svg)
    
    ![Divider Image](https://d3ddy8balm3goa.cloudfront.net/mlops-rag-workshop/divider-2.excalidraw.svg)
    ![Slide Two](https://d3ddy8balm3goa.cloudfront.net/rbc-llm-workshop/2.svg)
    
    ![Divider Image](https://d3ddy8balm3goa.cloudfront.net/mlops-rag-workshop/divider-2.excalidraw.svg)
    ![Slide Three](https://d3ddy8balm3goa.cloudfront.net/rbc-llm-workshop/3.svg)
    
    ![Divider Image](https://d3ddy8balm3goa.cloudfront.net/mlops-rag-workshop/divider-2.excalidraw.svg)
    ![Slide Four](https://d3ddy8balm3goa.cloudfront.net/rbc-llm-workshop/4-updated.svg)
    
    ## Example: A Gang of LLMs Tell A Story
    """
    logger.info("## Example: A Gang of LLMs Tell A Story")

    anthropic_llm = Anthropic(model="claude-3-opus-20240229")
    cohere_llm = Cohere(model="command")
    mistral_llm = MistralAI(model="mistral-large-latest")
    openai_llm = OllamaFunctionCallingAdapter(
        model="llama3.2", request_timeout=300.0, context_window=4096)

    start = anthropic_llm.complete(
        "Please start a random story. Limit your response to 20 words."
    )
    logger.debug(start)

    middle = cohere_llm.complete(
        f"Please continue the provided story. Limit your response to 20 words.\n\n {start.text}"
    )
    climax = mistral_llm.complete(
        f"Please continue the attached story. Your part is the climax of the story, so make it exciting! Limit your response to 20 words.\n\n {start.text + middle.text}"
    )
    ending = openai_llm.complete(
        f"Please continue the attached story. Your part is the end of the story, so wrap it up! Limit your response to 20 words.\n\n {start.text + middle.text + climax.text}"
    )

    logger.debug(f"{start}\n\n{middle}\n\n{climax}\n\n{ending}")

    """
    ![Divider Image](https://d3ddy8balm3goa.cloudfront.net/mlops-rag-workshop/divider-2.excalidraw.svg)
    ![Slide Five](https://d3ddy8balm3goa.cloudfront.net/rbc-llm-workshop/5.svg)
    
    ![Divider Image](https://d3ddy8balm3goa.cloudfront.net/mlops-rag-workshop/divider-2.excalidraw.svg)
    ![Slide Six](https://d3ddy8balm3goa.cloudfront.net/rbc-llm-workshop/6.svg)
    
    ## Example: Emergent Abilities (Zero Shot Classification)
    """
    logger.info("## Example: Emergent Abilities (Zero Shot Classification)")

    # import nest_asyncio

    # nest_asyncio.apply()

    sample_texts = [
        "Hey, friend! How are you today?",
        "Well, you're pretty crappy.",
    ]

    coros = []
    for txt in sample_texts:
        coro = openai_llm.acomplete(
            f"Classify the attached text as 'toxic' or 'not toxic'.\n\n{txt}"
        )
        coros.append(coro)
    classifications = await asyncio.gather(*coros)
    logger.success(format_json(classifications))

    [c.text for c in classifications]

    """
    ![Divider Image](https://d3ddy8balm3goa.cloudfront.net/mlops-rag-workshop/divider-2.excalidraw.svg)
    ![Slide Seven](https://d3ddy8balm3goa.cloudfront.net/rbc-llm-workshop/7.svg)
    
    ## Example: Chat Prompts
    """
    logger.info("## Example: Chat Prompts")

    chat_history_template = [
        ChatMessage(
            content="You are a helpful assistant that answers in the style of {style}",
            role=MessageRole.SYSTEM,
        ),
        ChatMessage(
            content="Tell me a short joke using 20 words.", role=MessageRole.USER
        ),
    ]
    chat_template = ChatPromptTemplate(chat_history_template)

    cohere_chat = Cohere(model="command-r-plus")

    shakespeare_response = cohere_chat.chat(
        messages=chat_template.format_messages(style="Shakespeare")
    )

    drake_response = cohere_chat.chat(
        messages=chat_template.format_messages(style="Drake")
    )

    logger.debug(shakespeare_response)

    logger.debug(drake_response)

    """
    ![Divider Image](https://d3ddy8balm3goa.cloudfront.net/mlops-rag-workshop/divider-2.excalidraw.svg)
    ![Slide Eight](https://d3ddy8balm3goa.cloudfront.net/rbc-llm-workshop/8.svg)
    
    ![Divider Image](https://d3ddy8balm3goa.cloudfront.net/mlops-rag-workshop/divider-2.excalidraw.svg)
    ![Slide Nine](https://d3ddy8balm3goa.cloudfront.net/rbc-llm-workshop/9.svg)
    
    ## Example: In-Context Learning And Chain Of Thought Prompting
    """
    logger.info("## Example: In-Context Learning And Chain Of Thought Prompting")

    qa_prompt_template = PromptTemplate(
        """
    You are a knowledgeable assistant able to perform arithmetic reasoning.
    
    {examples}
    
    {new_example}
    """
    )

    examples = """
    Q: Roger has 5 tennis balls. He buys 2 more cans of
    tennis balls. Each can has 3 tennis balls. How many
    tennis balls does he have now?
    
    A: Roger started with 5 balls. 2 cans of 3 tennis balls
    each is 6 tennis balls. 5 + 6 = 11. The answer is 11.
    """

    new_example = """
    Q: The cafeteria had 23 apples. If they used 20 to
    make lunch and bought 6 more, how many apples
    do they have?
    """

    prompt = qa_prompt_template.format(
        examples=examples, new_example=new_example)

    response = mistral_llm.complete(prompt)
    logger.debug(response)

    """
    ![Divider Image](https://d3ddy8balm3goa.cloudfront.net/mlops-rag-workshop/divider-2.excalidraw.svg)
    ![Slide Ten](https://d3ddy8balm3goa.cloudfront.net/rbc-llm-workshop/10-updated.svg)
    
    ## Example: Structured Data Extraction
    """
    logger.info("## Example: Structured Data Extraction")

    GOLF_CLUBS_LIST = Literal[
        "Driver",
        "Putter",
        "3wood",
        "SW",
    ]

    MISHIT_LIST = Literal[
        "Shank",
        "Fat",
        "Topped",
    ]

    class ShotRecord(BaseModel):
        """Data class for storing attributes of a golf shot."""

        club: GOLF_CLUBS_LIST = Field(
            description="The golf club used for the shot."
        )
        distance: int = Field(description="The distance the shot went")
        mishit: Optional[MISHIT_LIST] = Field(
            description="If the shot was mishit, then what kind of mishit. Default is None.",
            default=None,
        )
        on_target: bool = Field(
            description="Whether the shot was a good one and thus on target."
        )

    golf_shot_prompt_template = PromptTemplate(
        "Here is a description of a golf shot by the user. Please use it "
        "to record a data entry for this golf shot using the provided data class."
        "\n\n{shot_description}"
    )

    shot = openai_llm.structured_predict(
        output_cls=ShotRecord,
        prompt=golf_shot_prompt_template,
        shot_description="I hit my driver perfectly, 300 yards on the fairway",
    )
    logger.debug(shot)

    shot = openai_llm.structured_predict(
        output_cls=ShotRecord,
        prompt=golf_shot_prompt_template,
        shot_description="I duffed my sandwedge out of the sand, and it only went 5 yards.",
    )
    logger.debug(shot)

    """
    ## Notable Applications Powered By LLMs
    
    - ChatGPT
    - HuggingChat (Open-Source equivalent to ChatGPT)
    - Perplexity (Looking to overtake Google Search)
    
    ![Divider Image](https://d3ddy8balm3goa.cloudfront.net/mlops-rag-workshop/divider-2.excalidraw.svg)
    ![Slide Eleven](https://d3ddy8balm3goa.cloudfront.net/rbc-llm-workshop/11.svg)
    
    ![Divider Image](https://d3ddy8balm3goa.cloudfront.net/mlops-rag-workshop/divider-2.excalidraw.svg)
    ![Slide Twelve](https://d3ddy8balm3goa.cloudfront.net/rbc-llm-workshop/12.svg)
    
    ## Example: LLMs Lack Access To Updated Data
    """
    logger.info("## Notable Applications Powered By LLMs")

    response = mistral_llm.complete(
        "What can you tell me about the Royal Bank of Canada?"
    )

    logger.debug(response)

    query = "According to the 2023 Engagement Survey, what percentage of employees felt they contribute to RBC's success?"

    response = mistral_llm.complete(query)
    logger.debug(response)

    """
    ![Divider Image](https://d3ddy8balm3goa.cloudfront.net/mlops-rag-workshop/divider-2.excalidraw.svg)
    ![Slide Thirteen](https://d3ddy8balm3goa.cloudfront.net/rbc-llm-workshop/13-updated.svg)
    
    ## Example: RAG Yields More Accurate Responses
    """
    logger.info("## Example: RAG Yields More Accurate Responses")

    # !mkdir data
    # !wget "https://www.rbc.com/investor-relations/_assets-custom/pdf/ar_2023_e.pdf" -O f"{os.path.dirname(__file__)}/data/RBC-Annual-Report-2023.pdf"

    loader = SimpleDirectoryReader(
        input_dir=f"{os.path.dirname(__file__)}/data")
    documents = loader.load_data()
    index = VectorStoreIndex.from_documents(documents)
    rag = index.as_query_engine(llm=mistral_llm)

    response = rag.query(query)

    logger.debug(response)

    """
    ![Divider Image](https://d3ddy8balm3goa.cloudfront.net/mlops-rag-workshop/divider-2.excalidraw.svg)
    ![Slide Fourteen](https://d3ddy8balm3goa.cloudfront.net/rbc-llm-workshop/14.svg)
    
    ## Example: 3 Steps For Basic RAG (Unpacking the previous Example RAG)
    
    ### Step 1: Build Knowledge Store
    """
    logger.info(
        "## Example: 3 Steps For Basic RAG (Unpacking the previous Example RAG)")

    """Load the data.
    
    With llama-index, before any transformations are applied,
    data is loaded in the `Document` abstraction, which is
    a container that holds the text of the document.
    """

    loader = SimpleDirectoryReader(
        input_dir=f"{os.path.dirname(__file__)}/data")
    documents = loader.load_data()

    documents[1].text

    """Chunk, Encode, and Store into a Vector Store.
    
    To streamline the process, we can make use of the IngestionPipeline
    class that will apply your specified transformations to the
    Document's.
    """

    client = qdrant_client.QdrantClient(location=":memory:")
    vector_store = QdrantVectorStore(
        client=client, collection_name="test_store")

    pipeline = IngestionPipeline(
        transformations=[
            SentenceSplitter(),
            HuggingFaceEmbedding(
                model_name="sentence-transformers/all-MiniLM-L6-v2", cache_folder=MODELS_CACHE_DIR),
        ],
        vector_store=vector_store,
    )
    _nodes = pipeline.run(documents=documents, num_workers=4)

    """Create a llama-index... wait for it... Index.
    
    After uploading your encoded documents into your vector
    store of choice, you can connect to it with a VectorStoreIndex
    which then gives you access to all of the llama-index functionality.
    """

    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    """
    ### Step 2: Retrieve Against A Query
    """
    logger.info("### Step 2: Retrieve Against A Query")

    """Retrieve relevant documents against a query.
    
    With our Index ready, we can now query it to
    retrieve the most relevant document chunks.
    """

    retriever = index.as_retriever(similarity_top_k=2)
    retrieved_nodes = retriever.retrieve(query)

    retrieved_nodes

    """
    ### Step 3: Generate Final Response
    """
    logger.info("### Step 3: Generate Final Response")

    """Context-Augemented Generation.
    
    With our Index ready, we can create a QueryEngine
    that handles the retrieval and context augmentation
    in order to get the final response.
    """

    query_engine = index.as_query_engine(llm=mistral_llm)

    logger.debug(
        query_engine.get_prompts()[
            "response_synthesizer:text_qa_template"
        ].default_template.template
    )

    response = query_engine.query(query)
    logger.debug(response)

    """
    ![Divider Image](https://d3ddy8balm3goa.cloudfront.net/mlops-rag-workshop/divider-2.excalidraw.svg)
    ![Slide Fifteen](https://d3ddy8balm3goa.cloudfront.net/rbc-llm-workshop/15.svg)
    
    [Hi-Resolution Cheat Sheet](https://d3ddy8balm3goa.cloudfront.net/llamaindex/rag-cheat-sheet-final.svg)
    
    ![Divider Image](https://d3ddy8balm3goa.cloudfront.net/mlops-rag-workshop/divider-2.excalidraw.svg)
    ![Slide Sixteen](https://d3ddy8balm3goa.cloudfront.net/rbc-llm-workshop/16.svg)
    
    ![Divider Image](https://d3ddy8balm3goa.cloudfront.net/mlops-rag-workshop/divider-2.excalidraw.svg)
    ![Slide Seventeen](https://d3ddy8balm3goa.cloudfront.net/rbc-llm-workshop/17.svg)
    
    ![Divider Image](https://d3ddy8balm3goa.cloudfront.net/mlops-rag-workshop/divider-2.excalidraw.svg)
    ![Slide Eighteen](https://d3ddy8balm3goa.cloudfront.net/rbc-llm-workshop/18.svg)
    
    ## Example: Tool Use (or Function Calling)
    
    **Note:** LLMs are not very good pseudo-random number generators (see my [LinkedIn post](https://www.linkedin.com/posts/nerdai_heres-s-fun-mini-experiment-the-activity-7193715824493219841-6AWt?utm_source=share&utm_medium=member_desktop) about this)
    """
    logger.info("## Example: Tool Use (or Function Calling)")

    def uniform_random_sample(n: int) -> List[float]:
        """Generate a list a of uniform random numbers of size n between 0 and 1."""
        return random.rand(n).tolist()

    rs_tool = FunctionTool.from_defaults(fn=uniform_random_sample)

    agent = OllamaFunctionCallingAdapterAgent.from_tools(
        [rs_tool], llm=openai_llm, verbose=True)

    response = agent.chat(
        "Can you please give me a sample of 10 uniformly random numbers?"
    )
    logger.debug(str(response))

    """
    ![Divider Image](https://d3ddy8balm3goa.cloudfront.net/mlops-rag-workshop/divider-2.excalidraw.svg)
    ![Slide Nineteen](https://d3ddy8balm3goa.cloudfront.net/rbc-llm-workshop/19.svg)
    
    ## Example: Reflection Toxicity Reduction
    
    Here, we'll use llama-index `TollInteractiveReflectionAgent` to perform reflection and correction cycles on potentially harmful text. See the full demo [here](https://github.com/run-llama/llama_index/blob/main/docs/docs/examples/agent/introspective_agent_toxicity_reduction.ipynb).
    
    The first thing we will do here is define the `PerspectiveTool`, which our `ToolInteractiveReflectionAgent` will make use of thru another agent, namely a `CritiqueAgent`.
    
    To use Perspecive's API, you will need to do the following steps:
    
    1. Enable the Perspective API in your Google Cloud projects
    2. Generate a new set of credentials (i.e. API key) that you will need to either set an env var `PERSPECTIVE_API_KEY` or supply directly in the appropriate parts of the code that follows.
    
    To perform steps 1. and 2., you can follow the instructions outlined here: https://developers.perspectiveapi.com/s/docs-enable-the-api?language=en_US.
    
    ### Perspective API as Tool
    """
    logger.info("## Example: Reflection Toxicity Reduction")

    class Perspective:
        """Custom class to interact with Perspective API."""

        attributes = [
            "toxicity",
            "severe_toxicity",
            "identity_attack",
            "insult",
            "profanity",
            "threat",
            "sexually_explicit",
        ]

        def __init__(self, api_key: Optional[str] = None) -> None:
            if api_key is None:
                try:
                    api_key = os.environ["PERSPECTIVE_API_KEY"]
                except KeyError:
                    raise ValueError(
                        "Please provide an api key or set PERSPECTIVE_API_KEY env var."
                    )

            self._client = discovery.build(
                "commentanalyzer",
                "v1alpha1",
                developerKey=api_key,
                discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
                static_discovery=False,
            )

        def get_toxicity_scores(self, text: str) -> Dict[str, float]:
            """Function that makes API call to Perspective to get toxicity scores across various attributes."""
            analyze_request = {
                "comment": {"text": text},
                "requestedAttributes": {
                    att.upper(): {} for att in self.attributes
                },
            }

            response = (
                self._client.comments().analyze(body=analyze_request).execute()
            )
            try:
                return {
                    att: response["attributeScores"][att.upper()]["summaryScore"][
                        "value"
                    ]
                    for att in self.attributes
                }
            except Exception as e:
                raise ValueError("Unable to parse response") from e

    perspective = Perspective()

    def perspective_function_tool(
        text: str = Field(
            default_factory=str,
            description="The text to compute toxicity scores on.",
        ),
    ) -> Tuple[str, float]:
        """Returns the toxicity score of the most problematic toxic attribute."""
        scores = perspective.get_toxicity_scores(text=text)
        max_key = max(scores, key=scores.get)
        return (max_key, scores[max_key] * 100)

    pespective_tool = FunctionTool.from_defaults(
        perspective_function_tool,
    )

    perspective_function_tool(text="friendly greetings from python")

    """
    ### Build Agent To Reduce Toxicity of Harmful Text
    """
    logger.info("### Build Agent To Reduce Toxicity of Harmful Text")

    verbose = True
    critique_agent_worker = FunctionAgent(
        tools=[pespective_tool], llm=OllamaFunctionCallingAdapter("gpt-4.1")
    )
    correction_llm = OllamaFunctionCallingAdapter("gpt-4.1-mini")

    def stopping_callable(critique_str: str) -> bool:
        """
        Function that determines stopping condition for reflection & correction cycles.

        critique_str [str]: The response string provided by the critique agent.
        """
        return "[PASS]" in critique_str

    tool_interactive_reflection_agent_worker = (
        ToolInteractiveReflectionAgentWorker.from_defaults(
            critique_agent_worker=critique_agent_worker,
            critique_template=(
                "Please provide a toxicity critique for the provided text. "
                "If the toxicity score is less than 3.0 then please "
                "write '[PASS]' otherwise write '[FAIL]'. "
                "Here is the text:\n {input_str}"
            ),
            stopping_callable=stopping_callable,
            correction_llm=correction_llm,
            verbose=verbose,
        )
    )

    introspective_agent_worker = IntrospectiveAgentWorker.from_defaults(
        reflective_agent_worker=tool_interactive_reflection_agent_worker,
        verbose=verbose,
    )

    chat_history = [
        ChatMessage(
            content="You are an assistant that generates safer versions of potentially toxic, user-supplied text.",
            role=MessageRole.SYSTEM,
        )
    ]

    introspective_agent = introspective_agent_worker.as_agent(
        chat_history=chat_history, verbose=verbose
    )

    harmful_text = (
        "Those individuals who publicly tout eating no meat, "
        "especially when their stated reason has to do with caring "
        "about animals, are some of the most hypocritical people I’ve "
        "ever met. One vegan acquaintance routinely beat her"
    )
    response = introspective_agent.chat(harmful_text)

    logger.debug(response)

    """
    ![Divider Image](https://d3ddy8balm3goa.cloudfront.net/mlops-rag-workshop/divider-2.excalidraw.svg)
    ![Slide Twenty](https://d3ddy8balm3goa.cloudfront.net/rbc-llm-workshop/20.svg)
    
    ![Divider Image](https://d3ddy8balm3goa.cloudfront.net/mlops-rag-workshop/divider-2.excalidraw.svg)
    ![Slide TwentyOne](https://d3ddy8balm3goa.cloudfront.net/rbc-llm-workshop/21.svg)
    
    ![Divider Image](https://d3ddy8balm3goa.cloudfront.net/mlops-rag-workshop/divider-2.excalidraw.svg)
    ![Slide TwentyTwo](https://d3ddy8balm3goa.cloudfront.net/rbc-llm-workshop/22.svg)
    
    ![Divider Image](https://d3ddy8balm3goa.cloudfront.net/mlops-rag-workshop/divider-2.excalidraw.svg)
    ![Slide TwentyThree](https://d3ddy8balm3goa.cloudfront.net/rbc-llm-workshop/23.svg)
    
    ![Divider Image](https://d3ddy8balm3goa.cloudfront.net/mlops-rag-workshop/divider-2.excalidraw.svg)
    ![Slide TwentyFour](https://d3ddy8balm3goa.cloudfront.net/rbc-llm-workshop/24.svg)
    
    ## Example: LlamaReaders & LlamaPacks
    
    All of our integrations and packs can be discovered at [llamahub.ai](https://llamahub.ai). All of our packages/integrations are their own Python package that can be downloaded from PyPi.
    """
    logger.info("## Example: LlamaReaders & LlamaPacks")

    # %pip install llama-index-readers-wikipedia -q
    # %pip install wikipedia -q

    cities = ["Toronto", "Berlin", "Tokyo"]
    wiki_docs = WikipediaReader().load_data(pages=cities)

    wiki_docs[0].text[:500]

    """
    [Toronto Wikipedia Page](https://en.wikipedia.org/wiki/Toronto)
    
    ![Divider Image](https://d3ddy8balm3goa.cloudfront.net/mlops-rag-workshop/divider-2.excalidraw.svg)
    ![Slide TwentyFive](https://d3ddy8balm3goa.cloudfront.net/rbc-llm-workshop/25.svg)
    
    ![Divider Image](https://d3ddy8balm3goa.cloudfront.net/mlops-rag-workshop/divider-2.excalidraw.svg)
    ![Slide TwentySix](https://d3ddy8balm3goa.cloudfront.net/rbc-llm-workshop/26.svg)
    
    ![Divider Image](https://d3ddy8balm3goa.cloudfront.net/mlops-rag-workshop/divider-2.excalidraw.svg)
    ![Slide TwentySeven](https://d3ddy8balm3goa.cloudfront.net/rbc-llm-workshop/27-updated.svg)
    """

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
