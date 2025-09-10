from IPython.display import Image, display
from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Send
from pydantic import BaseModel, Field
from typing import List, Literal, Union, NamedTuple, Optional
from typing import Optional, Dict, Any
from typing_extensions import Annotated, TypedDict
import csv
import operator
import os
import requests
import shutil


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
# Tree of Thoughts

[Tree of Thoughts](https://arxiv.org/abs/2305.10601) (ToT), by Yao, et. al, is a general LLM agent search algorithm that combines reflection/evaluation and simple search (in this case BFS, though you can apply DFS or other algorithms if you'd like).

![LATS diagram](./img/tot.png)

It has three main steps:

1. Expand: generate 1 or more candidate solutions to the problem.
2. Score: measure the quality of the responses.
3. Prune: retain the top K best candidates

Then return to "Expand" if no solution is found (or if the solution is of insufficient quality).


## Prerequisites

We'll install the tutorial's dependent packages and set our API key for the LLM provider of choice.
"""
logger.info("# Tree of Thoughts")

# %%capture --no-stderr
# %pip install -U langgraph langchain-ollama

# import getpass


def _set_env(var: str):
    if not os.environ.get(var):
#         os.environ[var] = getpass.getpass(f"{var}: ")


# _set_env("OPENAI_API_KEY")
trace = True
if trace:
    _set_env("LANGSMITH_API_KEY")
    os.environ["LANGSMITH_PROJECT"] = "ToT Tutorial"

"""
## Task Definition

Our agent will try to play the "Game of 24". Given 4 numbers, it must generate a math equation that uses each of these numbers exactly one time to evaluate to a value of `24`.
"""
logger.info("## Task Definition")


OperatorType = Literal["+", "-", "*", "/"]
TokenType = Union[float, OperatorType]



class Equation(BaseModel):
    """The formula combining the provided numbers to reach the target of 24."""

    tokens: List[TokenType] = Field(
        description="The stack of tokens and operators in reverse-polish notation. Example: [3, 4, '+', -1, '*'] would evaluate to (3 + 4) * -1 = -7.",
    )

    def compute(self) -> float:
        op_funcs = {
            "+": operator.add,
            "-": operator.sub,
            "*": operator.mul,
            "/": operator.truediv,
        }
        stack = []
        for token in self.tokens:
            if isinstance(token, float):
                stack.append(token)
            else:
                b, a = stack.pop(), stack.pop()
                stack.append(op_funcs[token](a, b))

        return stack[0]


class GuessEquations(BaseModel):
    """Submit multiple equations as guesses."""

    reasoning: str = Field(
        description="The reasoning behind the submitted guesses. Explain how you arrived at these equations."
    )

    equations: List[Equation] = Field(
        description="The list of equations to submit as guesses."
    )




class Candidate(NamedTuple):
    candidate: Equation
    score: Optional[float] = None
    feedback: Optional[str] = None

    def __str__(self):
        try:
            computed = self.candidate.compute()
        except Exception as e:
            computed = f"Invalid equation: {self.candidate.tokens}; Error: {repr(e)}"

        return f"Equation({self.candidate.tokens}) = {computed} (Reward: {self.score})"


class ScoredCandidate(Candidate):
    candidate: Equation
    score: float
    feedback: str

"""
#### Fetch data

We'll use an example from the [Game of 24](https://github.com/princeton-nlp/tree-of-thought-llm) dataset.
"""
logger.info("#### Fetch data")


csv_data = requests.get(
    "https://storage.googleapis.com/benchmarks-artifacts/game-of-24/24.csv"
).content.decode("utf-8")
puzzles = [row[1].strip() for row in csv.reader(csv_data.splitlines()[1:])]

logger.debug(f"Example puzzles: {puzzles[:3]}")

"""
## Expander

The "tree of thoughts" algorithm is relatively generic. The primary two task-specific components are the **expander** and the **scorer**.
The expander (the augmented LLM) tries to generate 1 or more solutions to the problem. On subsequent attempts, it is given a seed/candidate value from 
the previous search.

You can update this section to match your own task requirements. The expander can be arbitrarily complex. All that's required is that it accepts the problem and an optional previous attempt (or attempts) and returns a new result.
"""
logger.info("## Expander")



prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are playing the Game of 24. Using the provide numbers, create an equation that evaluates to 24.\n"
            "Submit exactly {k} guesses for this round.",
        ),
        ("user", "Solve the 24 game for these numbers: {problem}.{candidate}"),
    ],
).partial(candidate="")
llm = ChatOllama(model="llama3.2")

bound_llm = llm.with_structured_output(GuessEquations)
solver = prompt | bound_llm

"""
# Scorer

In this game, the scorer is easy. We need to assert two things:

1. The LLM has generated a valid equation using each number exactly one time.
2. The equation evaluates to 24.

You can update this function to match your own task requirements.
"""
logger.info("# Scorer")

def compute_score(problem: str, candidate: Candidate) -> ScoredCandidate:
    numbers = list(map(int, problem.split()))
    used_numbers = [
        token for token in candidate.candidate.tokens if isinstance(token, float)
    ]
    if sorted(used_numbers) != sorted(numbers):
        score = 0
        feedback = "The equation must use all 4 numbers exactly once."
        return ScoredCandidate(
            candidate=candidate.candidate, score=score, feedback=feedback
        )
    try:
        result = candidate.candidate.compute()
        score = 1 / (1 + abs(24 - result))
        feedback = f"Result: {result}"
    except Exception as e:
        score = 0
        feedback = f"Invalid equation. Error: {repr(e)}"
    return ScoredCandidate(
        candidate=candidate.candidate, score=score, feedback=feedback
    )

"""
## Graph

Now it's time to create our graph.
"""
logger.info("## Graph")




def update_candidates(
    existing: Optional[list] = None,
    updates: Optional[Union[list, Literal["clear"]]] = None,
) -> List[str]:
    if existing is None:
        existing = []
    if updates is None:
        return existing
    if updates == "clear":
        return []
    return existing + updates


class ToTState(TypedDict):
    problem: str
    candidates: Annotated[List[Candidate], update_candidates]
    scored_candidates: Annotated[List[ScoredCandidate], update_candidates]
    depth: Annotated[int, operator.add]


class Context(TypedDict, total=False):
    max_depth: int
    threshold: float
    k: int
    beam_size: int


class EnsuredContext(TypedDict):
    max_depth: int
    threshold: float
    k: int
    beam_size: int


def _ensure_context(ctx: Context) -> EnsuredContext:
    """Get params that configure the search algorithm."""
    return {
        "max_depth": ctx.get("max_depth", 10),
        "threshold": ctx.get("threshold", 0.9),
        "k": ctx.get("k", 5),
        "beam_size": ctx.get("beam_size", 3),
    }


class ExpansionState(ToTState):
    seed: Optional[Candidate]


def expand(
    state: ExpansionState, *, runtime: Runtime[Context]
) -> Dict[str, List[Candidate]]:
    """Generate the next state."""
    ctx = _ensure_context(runtime.context)
    if not state.get("seed"):
        candidate_str = ""
    else:
        candidate_str = "\n\n" + str(state["seed"])
    try:
        equation_submission = solver.invoke(
            {
                "problem": state["problem"],
                "candidate": candidate_str,
                "k": ctx["k"],
            },
        )
    except Exception:
        return {"candidates": []}
    new_candidates = [
        Candidate(candidate=equation) for equation in equation_submission.equations
    ]
    return {"candidates": new_candidates}


def score(state: ToTState) -> Dict[str, Any]:
    """Evaluate the candidate generations."""
    candidates = state["candidates"]
    scored = []
    for candidate in candidates:
        scored.append(compute_score(state["problem"], candidate))
    return {"scored_candidates": scored, "candidates": "clear"}


def prune(state: ToTState, *, runtime: Runtime[Context]) -> Dict[str, Any]:
    scored_candidates = state["scored_candidates"]
    beam_size = _ensure_context(runtime.context)["beam_size"]
    organized = sorted(
        scored_candidates, key=lambda candidate: candidate[1], reverse=True
    )
    pruned = organized[:beam_size]
    return {
        "candidates": pruned,
        "scored_candidates": "clear",
        "depth": 1,
    }


def should_terminate(
    state: ToTState, runtime: Runtime[Context]
) -> Union[Literal["__end__"], Send]:
    ctx = _ensure_context(runtime.context)
    solved = state["candidates"][0].score >= ctx["threshold"]
    if solved or state["depth"] >= ctx["max_depth"]:
        return "__end__"
    return [
        Send("expand", {**state, "somevalseed": candidate})
        for candidate in state["candidates"]
    ]


builder = StateGraph(state_schema=ToTState, context_schema=Context)

builder.add_node(expand)
builder.add_node(score)
builder.add_node(prune)

builder.add_edge("expand", "score")
builder.add_edge("score", "prune")
builder.add_conditional_edges("prune", should_terminate, path_map=["expand", "__end__"])

builder.add_edge("__start__", "expand")

graph = builder.compile(checkpointer=InMemorySaver())


display(Image(graph.get_graph().draw_mermaid_png()))

"""
## Run

Now let's try it on one of the puzzles!
"""
logger.info("## Run")

for step in graph.stream(
    {"problem": puzzles[42]},
    config={"configurable": {"thread_id": "test_1"}},
    context={"depth": 10},
):
    logger.debug(step)

final_state = graph.get_state({"configurable": {"thread_id": "test_1"}})
winning_solution = final_state.values["candidates"][0]
search_depth = final_state.values["depth"]
if winning_solution[1] == 1:
    logger.debug(f"Found a winning solution in {search_depth} steps: {winning_solution}")
else:
    logger.debug(
        f"Failed to find a winning solution in {search_depth} steps. Best guess: {winning_solution}"
    )

logger.info("\n\n[DONE]", bright=True)