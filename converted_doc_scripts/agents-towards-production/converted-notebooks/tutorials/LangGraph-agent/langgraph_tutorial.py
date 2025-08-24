from IPython.display import Image, display
from IPython.display import display, Image
from dotenv import load_dotenv
from jet.llm.mlx.base_langchain import ChatMLX
from jet.logger import CustomLogger
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_core.runnables.graph import MermaidDrawMethod
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Annotated
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
![](https://europe-west1-atp-views-tracker.cloudfunctions.net/working-analytics?notebook=tutorials--langgraph-agent--langgraph-tutorial)

# LangGraph Tutorial: Building a Text Analysis Pipeline

## Introduction to LangGraph

LangGraph is a powerful framework by LangChain designed for creating stateful, multi-actor applications with LLMs. It provides the structure and tools needed to build sophisticated AI agents through a graph-based approach.

Think of LangGraph as an architect's drafting table - it gives us the tools to design how our agent will think and act. Just as an architect draws blueprints showing how different rooms connect and how people will flow through a building, LangGraph lets us design how different capabilities will connect and how information will flow through our agent.

### Key Features:
- **State Management:** Maintain persistent state across interactions
- **Flexible Routing:** Define complex flows between components
- **Persistence:** Save and resume workflows
- **Visualization:** See and understand your agent's structure

In this tutorial, we'll demonstrate LangGraph by building a multi-step text analysis pipeline that processes text through three stages:
1. **Text Classification:** Categorize input text into predefined categories
2. **Entity Extraction:** Identify key entities from the text
3. **Text Summarization:** Generate a concise summary of the input text

This pipeline showcases how LangGraph can be used to create a modular, extensible workflow for natural language processing tasks.

📖 **For more background on building AI agents and getting started, check out our detailed blog post:** [Your First AI Agent: Simpler Than You Think](https://open.substack.com/pub/diamantai/p/your-first-ai-agent-simpler-than?r=336pe4&utm_campaign=post&utm_medium=web&showWelcomeOnShare=false)

*DiamantAI is a top 0.1% newsletter for AI with over 25,000 subscribers, focusing on AI techniques, breakthroughs, and tutorials.*

---

## Setting Up Our Environment

Before diving into the code, let's set up our development environment.
"""
logger.info("# LangGraph Tutorial: Building a Text Analysis Pipeline")

# !pip install langgraph langchain langchain-openai python-dotenv

"""
### Setting Up API Keys

We'll need an MLX API key to use their models. If you haven't already, you can get one from [https://platform.openai.com/signup](https://platform.openai.com/signup).

Let's load our environment variables:
"""
logger.info("### Setting Up API Keys")


load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

"""
### Testing Our Setup

Let's make sure our environment is working correctly by creating a simple test with the MLX model:
"""
logger.info("### Testing Our Setup")


llm = ChatMLX(model="qwen3-1.7b-4bit-mini")

response = llm.invoke("Hello! Are you working?")
logger.debug(response.content)

"""
## Building Our Text Analysis Pipeline

Now let's import the necessary packages for our LangGraph text analysis pipeline:
"""
logger.info("## Building Our Text Analysis Pipeline")


"""
### Designing Our Agent's Memory

Just as human intelligence requires memory, our agent needs a way to keep track of information. We create this using a TypedDict to define our state structure:
"""
logger.info("### Designing Our Agent's Memory")

class State(TypedDict):
    text: str
    classification: str
    entities: List[str]
    summary: str

llm = ChatMLX(model="qwen3-1.7b-4bit-mini", temperature=0)

"""
### Creating Our Agent's Core Capabilities

Now we'll create the actual skills our agent will use. Each of these capabilities is implemented as a function that performs a specific type of analysis.

#### 1. Classification Node
"""
logger.info("### Creating Our Agent's Core Capabilities")

def classification_node(state: State):
    '''Classify the text into one of the categories: News, Blog, Research, or Other'''
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Classify the following text into one of the categories: News, Blog, Research, or Other.\n\nText:{text}\n\nCategory:"
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    classification = llm.invoke([message]).content.strip()
    return {"classification": classification}

"""
#### 2. Entity Extraction Node
"""
logger.info("#### 2. Entity Extraction Node")

def entity_extraction_node(state: State):
    '''Extract all the entities (Person, Organization, Location) from the text'''
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Extract all the entities (Person, Organization, Location) from the following text. Provide the result as a comma-separated list.\n\nText:{text}\n\nEntities:"
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    entities = llm.invoke([message]).content.strip().split(", ")
    return {"entities": entities}

"""
#### 3. Summarization Node
"""
logger.info("#### 3. Summarization Node")

def summarization_node(state: State):
    '''Summarize the text in one short sentence'''
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following text in one short sentence.\n\nText:{text}\n\nSummary:"
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    summary = llm.invoke([message]).content.strip()
    return {"summary": summary}

"""
### Bringing It All Together

Now comes the most exciting part - connecting these capabilities into a coordinated system using LangGraph:
"""
logger.info("### Bringing It All Together")

workflow = StateGraph(State)

workflow.add_node("classification_node", classification_node)
workflow.add_node("entity_extraction", entity_extraction_node)
workflow.add_node("summarization", summarization_node)

workflow.set_entry_point("classification_node")  # Set the entry point of the graph
workflow.add_edge("classification_node", "entity_extraction")
workflow.add_edge("entity_extraction", "summarization")
workflow.add_edge("summarization", END)

app = workflow.compile()

"""
### Visualizing the Workflow

One of the powerful features of LangGraph is the ability to visualize our workflow:
"""
logger.info("### Visualizing the Workflow")

try:
    display(
        Image(
            app.get_graph().draw_mermaid_png(
                draw_method=MermaidDrawMethod.API,
            )
        )
    )
except Exception as e:
    logger.debug(f"Error generating visualization: {e}")
    logger.debug("The graph structure is: classification_node -> entity_extraction -> summarization -> END")

"""
## Testing Our Agent

Now that we've built our agent, let's see how it performs with a real-world text example:
"""
logger.info("## Testing Our Agent")

sample_text = """
MLX has announced the GPT-4 model, which is a large multimodal model that exhibits human-level performance on various professional benchmarks. It is developed to improve the alignment and safety of AI systems.
Additionally, the model is designed to be more efficient and scalable than its predecessor, GPT-3. The GPT-4 model is expected to be released in the coming months and will be available to the public for research and development purposes.
"""

state_input = {"text": sample_text}
result = app.invoke(state_input)

logger.debug("Classification:", result["classification"])
logger.debug("\nEntities:", result["entities"])
logger.debug("\nSummary:", result["summary"])

"""
## Understanding the Power of Coordinated Processing

What makes this result particularly impressive isn't just the individual outputs - it's how each step builds on the others to create a complete understanding of the text.

- The **classification** provides context that helps frame our understanding of the text type
- The **entity extraction** identifies important names and concepts
- The **summarization** distills the essence of the document

This mirrors human reading comprehension, where we naturally form an understanding of what kind of text it is, note important names and concepts, and form a mental summary - all while maintaining the relationships between these different aspects of understanding.

## Try with Your Own Text

Now let's try our pipeline with another text sample of your choice:
"""
logger.info("## Understanding the Power of Coordinated Processing")

your_text = """
The recent advancements in quantum computing have opened new possibilities for cryptography and data security.
Researchers at MIT and Google have demonstrated quantum algorithms that could potentially break current encryption methods.
However, they are also developing new quantum-resistant encryption techniques to protect data in the future.
"""

your_result = app.invoke({"text": your_text})

logger.debug("Classification:", your_result["classification"])
logger.debug("\nEntities:", your_result["entities"])
logger.debug("\nSummary:", your_result["summary"])

"""
## Adding More Capabilities (Advanced)

One of the powerful aspects of LangGraph is how easily we can extend our agent with new capabilities. Let's add a sentiment analysis node to our pipeline:
"""
logger.info("## Adding More Capabilities (Advanced)")

class EnhancedState(TypedDict):
    text: str
    classification: str
    entities: List[str]
    summary: str
    sentiment: str

def sentiment_node(state: EnhancedState):
    '''Analyze the sentiment of the text: Positive, Negative, or Neutral'''
    prompt = PromptTemplate(
        input_variables=["text"],
        template="Analyze the sentiment of the following text. Is it Positive, Negative, or Neutral?\n\nText:{text}\n\nSentiment:"
    )
    message = HumanMessage(content=prompt.format(text=state["text"]))
    sentiment = llm.invoke([message]).content.strip()
    return {"sentiment": sentiment}

enhanced_workflow = StateGraph(EnhancedState)

enhanced_workflow.add_node("classification_node", classification_node)
enhanced_workflow.add_node("entity_extraction", entity_extraction_node)
enhanced_workflow.add_node("summarization", summarization_node)

enhanced_workflow.add_node("sentiment_analysis", sentiment_node)

enhanced_workflow.set_entry_point("classification_node")
enhanced_workflow.add_edge("classification_node", "entity_extraction")
enhanced_workflow.add_edge("entity_extraction", "summarization")
enhanced_workflow.add_edge("summarization", "sentiment_analysis")
enhanced_workflow.add_edge("sentiment_analysis", END)

enhanced_app = enhanced_workflow.compile()

"""
### Testing the Enhanced Agent
"""
logger.info("### Testing the Enhanced Agent")

enhanced_result = enhanced_app.invoke({"text": sample_text})

logger.debug("Classification:", enhanced_result["classification"])
logger.debug("\nEntities:", enhanced_result["entities"])
logger.debug("\nSummary:", enhanced_result["summary"])
logger.debug("\nSentiment:", enhanced_result["sentiment"])

"""
## Adding Conditional Edges to the LangGraph Pipeline(Advanced Logic)

### Why **Conditional Edges**?

So far, our graph has followed a fixed linear path:

classification_node → entity_extraction → summarization → (sentiment)

But in real-world applications, we often want to run certain steps only if needed. For example:

- Only extract entities if the text is a News or Research article.

- Skip summarization if the text is very short.

- Add custom processing for Blog posts.

LangGraph makes this easy through conditional edges-logic gates that dynamically route execution based on data in the current state.

### 1. Update the State
We'll use the same EnhancedState from the advanced section:
"""
logger.info("## Adding Conditional Edges to the LangGraph Pipeline(Advanced Logic)")

class EnhancedState(TypedDict):
    text: str
    classification: str
    entities: List[str]
    summary: str
    sentiment: str

"""
### 2. Create a Routing Function
This function routes based on classification:
"""
logger.info("### 2. Create a Routing Function")

def route_after_classification(state: EnhancedState) -> str:
    category = state["classification"].lower() # returns: "news", "blog", "research", "other"
    return category in ["news", "research"]

"""
### 3. Define the Graph
"""
logger.info("### 3. Define the Graph")


conditional_workflow = StateGraph(EnhancedState)

conditional_workflow.add_node("classification_node", classification_node)
conditional_workflow.add_node("entity_extraction", entity_extraction_node)
conditional_workflow.add_node("summarization", summarization_node)
conditional_workflow.add_node("sentiment_analysis", sentiment_node)

conditional_workflow.set_entry_point("classification_node")

conditional_workflow.add_conditional_edges("classification_node", route_after_classification, path_map={
    True: "entity_extraction",
    False: "summarization"
})

conditional_workflow.add_edge("entity_extraction", "summarization")
conditional_workflow.add_edge("summarization", "sentiment_analysis")
conditional_workflow.add_edge("sentiment_analysis", END)

conditional_app = conditional_workflow.compile()

"""
### 4. Visualize
"""
logger.info("### 4. Visualize")


try:
    display(Image(
        conditional_app.get_graph().draw_mermaid_png()
    ))
except:
    logger.debug("Graph: classification_node → (conditional) → [entity_extraction or summarization] → sentiment_analysis → END")

"""
### 5. Test the Conditional Pipeline
"""
logger.info("### 5. Test the Conditional Pipeline")

test_text = """
MLX released the GPT-4 model with enhanced performance on academic and professional tasks. It's seen as a major breakthrough in alignment and reasoning capabilities.
"""

result = conditional_app.invoke({"text": test_text})

logger.debug("Classification:", result["classification"])
logger.debug("Entities:", result.get("entities", "Skipped"))
logger.debug("Summary:", result["summary"])
logger.debug("Sentiment:", result["sentiment"])

"""
Now try it with a Blog:
"""
logger.info("Now try it with a Blog:")

blog_text = """
Here's what I learned from a week of meditating in silence. No phones, no talking—just me, my breath, and some deep realizations.
"""

result = conditional_app.invoke({"text": blog_text})

logger.debug("Classification:", result["classification"])
logger.debug("Entities:", result.get("entities", "Skipped (not applicable)"))
logger.debug("Summary:", result["summary"])
logger.debug("Sentiment:", result["sentiment"])

"""
With conditional edges, our agent can now:

- Make decisions based on context

- Skip unnecessary steps

- Run faster and cheaper

- Behave more intelligently

## Conclusion

In this tutorial, we've:

1. Explored LangGraph concepts and its graph-based approach
2. Built a text processing pipeline with classification, entity extraction, and summarization
3. Enhanced our pipeline with additional capabilities
4. Introduced conditional edges to dynamically control the flow based on classification results
5. Visualized our workflow
6. Tested our agent with real-world text examples

LangGraph provides a powerful framework for creating AI agents by modeling them as graphs of capabilities. This approach makes it easy to design, modify, and extend complex AI systems.

### Next Steps

- Add more nodes to extend your agent's capabilities
- Experiment with different LLMs and parameters
- Explore LangGraph's state persistence features for ongoing conversations

For more information, check out the official LangGraph documentation at https://langchain-ai.github.io/langgraph/


"""
logger.info("## Conclusion")

logger.info("\n\n[DONE]", bright=True)