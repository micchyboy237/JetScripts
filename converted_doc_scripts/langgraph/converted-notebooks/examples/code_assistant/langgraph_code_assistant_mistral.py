from IPython.display import Image, display
from jet.logger import CustomLogger
from langchain_core.pydantic_v1 import BaseModel, Field
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import AnyMessage, add_messages
from typing import Annotated, TypedDict
import os
import shutil
import uuid


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
# Code generation with self-correction

AlphaCodium presented an approach for code generation that uses control flow.

Main idea: [construct an answer to a coding question iteratively.](https://x.com/karpathy/status/1748043513156272416?s=20). 

[AlphaCodium](https://github.com/Codium-ai/AlphaCodium) iteravely tests and improves an answer on public and AI-generated tests for a particular question. 

We will implement some of these ideas from scratch using [LangGraph](https://langchain-ai.github.io/langgraph/):

1. We show how to route user questions to different types of documentation
2. We we will show how to perform inline unit tests to confirm imports and code execution work
3. We will show how to use LangGraph to orchestrate this

![Screenshot 2024-05-23 at 2.17.51 PM.png](attachment:15d3ac32-cdf3-4800-a30c-f26d828d69c8.png)
"""
logger.info("# Code generation with self-correction")

# ! pip install -U langchain_community langchain-mistralai langchain langgraph

"""
### LLM

We'll use the Mistral API and `Codestral` instruct model, which support tool use!
"""
logger.info("### LLM")


os.environ["TOKENIZERS_PARALLELISM"] = "true"
mistral_api_key = os.getenv("MISTRAL_API_KEY")  # Ensure this is set

"""
### Tracing

Optionally, we'll use LangSmith for tracing.
"""
logger.info("### Tracing")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "<your-api-key>"
os.environ["LANGCHAIN_PROJECT"] = "Mistral-code-gen-testing"

"""
## Code Generation

Test with structured output.
"""
logger.info("## Code Generation")



question = "Write a function for fibonacci."
messages = [("user", question)]

"""
## State
"""
logger.info("## State")




class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        error : Binary flag for control flow to indicate whether test error was tripped
        messages : With user question, error messages, reasoning
        generation : Code solution
        iterations : Number of tries
    """

    error: str
    messages: Annotated[list[AnyMessage], add_messages]
    generation: str
    iterations: int

"""
## Graph
"""
logger.info("## Graph")



### Parameters
max_iterations = 3


### Nodes
def generate(state: GraphState):
    """
    Generate a code solution

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation
    """

    logger.debug("---GENERATING CODE SOLUTION---")

    # State
    messages = state["messages"]
    iterations = state["iterations"]

    # Solution
    code_solution = code_gen_chain.invoke(messages)
    messages += [
        (
            "assistant",
            f"Here is my attempt to solve the problem: {code_solution.prefix} \n Imports: {code_solution.imports} \n Code: {code_solution.code}",
        )
    ]

    # Increment
    iterations = iterations + 1
    return {"generation": code_solution, "messages": messages, "iterations": iterations}


def code_check(state: GraphState):
    """
    Check code

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, error
    """

    logger.debug("---CHECKING CODE---")

    # State
    messages = state["messages"]
    code_solution = state["generation"]
    iterations = state["iterations"]

    # Get solution components
    imports = code_solution.imports
    code = code_solution.code

    # Check imports
    try:
        exec(imports)
    except Exception as e:
        logger.debug("---CODE IMPORT CHECK: FAILED---")
        error_message = [
            (
                "user",
                f"Your solution failed the import test. Here is the error: {e}. Reflect on this error and your prior attempt to solve the problem. (1) State what you think went wrong with the prior solution and (2) try to solve this problem again. Return the FULL SOLUTION. Use the code tool to structure the output with a prefix, imports, and code block:",
            )
        ]
        messages += error_message
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "yes",
        }

    # Check execution
    try:
        combined_code = f"{imports}\n{code}"
        logger.debug(f"CODE TO TEST: {combined_code}")
        # Use a shared scope for exec
        global_scope = {}
        exec(combined_code, global_scope)
    except Exception as e:
        logger.debug("---CODE BLOCK CHECK: FAILED---")
        error_message = [
            (
                "user",
                f"Your solution failed the code execution test: {e}) Reflect on this error and your prior attempt to solve the problem. (1) State what you think went wrong with the prior solution and (2) try to solve this problem again. Return the FULL SOLUTION. Use the code tool to structure the output with a prefix, imports, and code block:",
            )
        ]
        messages += error_message
        return {
            "generation": code_solution,
            "messages": messages,
            "iterations": iterations,
            "error": "yes",
        }

    # No errors
    logger.debug("---NO CODE TEST FAILURES---")
    return {
        "generation": code_solution,
        "messages": messages,
        "iterations": iterations,
        "error": "no",
    }


### Conditional edges


def decide_to_finish(state: GraphState):
    """
    Determines whether to finish.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
    error = state["error"]
    iterations = state["iterations"]

    if error == "no" or iterations == max_iterations:
        logger.debug("---DECISION: FINISH---")
        return "end"
    else:
        logger.debug("---DECISION: RE-TRY SOLUTION---")
        return "generate"


### Utilities


def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        logger.debug("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            logger.debug(msg_repr)
            _printed.add(message.id)


builder = StateGraph(GraphState)

# Define the nodes
builder.add_node("generate", generate)  # generation solution
builder.add_node("check_code", code_check)  # check code

# Build graph
builder.add_edge(START, "generate")
builder.add_edge("generate", "check_code")
builder.add_conditional_edges(
    "check_code",
    decide_to_finish,
    {
        "end": END,
        "generate": "generate",
    },
)

memory = InMemorySaver()
graph = builder.compile(checkpointer=memory)


try:
    display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

_printed = set()
thread_id = str(uuid.uuid4())
config = {
    "configurable": {
        # Checkpoints are accessed by thread_id
        "thread_id": thread_id,
    }
}

question = "Write a Python program that prints 'Hello, World!' to the console."
events = graph.stream(
    {"messages": [("user", question)], "iterations": 0}, config, stream_mode="values"
)
for event in events:
    _print_event(event, _printed)

"""
`Trace:`

https://smith.langchain.com/public/53bcdaab-e3c5-4423-9908-c44595325c38/r
"""
logger.info("https://smith.langchain.com/public/53bcdaab-e3c5-4423-9908-c44595325c38/r")

_printed = set()
thread_id = str(uuid.uuid4())
config = {
    "configurable": {
        # Checkpoints are accessed by thread_id
        "thread_id": thread_id,
    }
}

question = """Create a Python program that checks if a given string is a palindrome. A palindrome is a word, phrase, number, or other sequence of characters that reads the same forward and backward (ignoring spaces, punctuation, and capitalization).

Requirements:
The program should define a function is_palindrome(s) that takes a string s as input.
The function should return True if the string is a palindrome and False otherwise.
Ignore spaces, punctuation, and case differences when checking for palindromes.

Give an example of it working on an example input word."""

events = graph.stream(
    {"messages": [("user", question)], "iterations": 0}, config, stream_mode="values"
)
for event in events:
    _print_event(event, _printed)

"""
Trace:

https://smith.langchain.com/public/e749936d-7746-49de-b980-c41b17986e79/r
"""
logger.info("Trace:")

_printed = set()
thread_id = str(uuid.uuid4())
config = {
    "configurable": {
        # Checkpoints are accessed by thread_id
        "thread_id": thread_id,
    }
}

question = """Write a program that prints the numbers from 1 to 100.
But for multiples of three, print "Fizz" instead of the number, and for the multiples of five, print "Buzz".
For numbers which are multiples of both three and five, print "FizzBuzz"."""

events = graph.stream(
    {"messages": [("user", question)], "iterations": 0}, config, stream_mode="values"
)
for event in events:
    _print_event(event, _printed)

"""
Trace: 

https://smith.langchain.com/public/f5c19708-7592-4512-9f00-9696ab34a9eb/r
"""
logger.info("Trace:")


_printed = set()
thread_id = str(uuid.uuid4())
config = {
    "configurable": {
        # Checkpoints are accessed by thread_id
        "thread_id": thread_id,
    }
}

question = """I want to vectorize a function

        frame = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        for i, val1 in enumerate(rows):
            for j, val2 in enumerate(cols):
                for j, val3 in enumerate(ch):
                    # Assuming you want to store the pair as tuples in the matrix
                    frame[i, j, k] = image[val1, val2, val3]

        out.write(np.array(frame))

with a simple numpy function that does something like this what is it called. Show me a test case with this working."""

events = graph.stream(
    {"messages": [("user", question)], "iterations": 0}, config, stream_mode="values"
)
for event in events:
    _print_event(event, _printed)

"""
Trace w/ good example of self-correction:

https://smith.langchain.com/public/b54778a0-d267-4f09-bc28-71761201c522/r
"""
logger.info("Trace w/ good example of self-correction:")

_printed = set()
thread_id = str(uuid.uuid4())
config = {
    "configurable": {
        # Checkpoints are accessed by thread_id
        "thread_id": thread_id,
    }
}

question = """Create a Python program that allows two players to play a game of Tic-Tac-Toe. The game should be played on a 3x3 grid. The program should:

- Allow players to take turns to input their moves.
- Check for invalid moves (e.g., placing a marker on an already occupied space).
- Determine and announce the winner or if the game ends in a draw.

Requirements:
- Use a 2D list to represent the Tic-Tac-Toe board.
- Use functions to modularize the code.
- Validate player input.
- Check for win conditions and draw conditions after each move."""

events = graph.stream(
    {"messages": [("user", question)], "iterations": 0}, config, stream_mode="values"
)
for event in events:
    _print_event(event, _printed)

"""
Trace w/ good example of failure to correct:

https://smith.langchain.com/public/871ae736-2f77-44d4-b0da-a600d8f5377d/r
"""
logger.info("Trace w/ good example of failure to correct:")


logger.info("\n\n[DONE]", bright=True)