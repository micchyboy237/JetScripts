from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain_experimental.tot.base import ToTChain
from langchain_experimental.tot.checker import ToTChecker
from langchain_experimental.tot.thought import ThoughtValidity
from typing import Tuple
import os
import re
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
# Tree of Thought (ToT) example

The Tree of Thought (ToT) is a chain that allows you to query a Large Language Model (LLM) using the Tree of Thought technique. This is based on the paper ["Large Language Model Guided Tree-of-Thought"](https://arxiv.org/pdf/2305.08291.pdf)
"""
logger.info("# Tree of Thought (ToT) example")


llm = ChatOllama(temperature=1, max_tokens=512, model="llama3.2")

sudoku_puzzle = "3,*,*,2|1,*,3,*|*,1,*,3|4,*,*,1"
sudoku_solution = "3,4,1,2|1,2,3,4|2,1,4,3|4,3,2,1"
problem_description = f"""
{sudoku_puzzle}

- This is a 4x4 Sudoku puzzle.
- The * represents a cell to be filled.
- The | character separates rows.
- At each step, replace one or more * with digits 1-4.
- There must be no duplicate digits in any row, column or 2x2 subgrid.
- Keep the known digits from previous valid thoughts in place.
- Each thought can be a partial or the final solution.
""".strip()
logger.debug(problem_description)

"""
## Rules Based Checker

Each thought is evaluated by the thought checker and is given a validity type: valid, invalid or partial. A simple checker can be rule based. For example, in the case of a sudoku puzzle, the checker can check if the puzzle is valid, invalid or partial.

In the following code we implement a simple rule based checker for a specific 4x4 sudoku puzzle.
"""
logger.info("## Rules Based Checker")




class MyChecker(ToTChecker):
    def evaluate(
        self, problem_description: str, thoughts: Tuple[str, ...] = ()
    ) -> ThoughtValidity:
        last_thought = thoughts[-1]
        clean_solution = last_thought.replace(" ", "").replace('"', "")
        regex_solution = clean_solution.replace("*", ".").replace("|", "\\|")
        if sudoku_solution in clean_solution:
            return ThoughtValidity.VALID_FINAL
        elif re.search(regex_solution, sudoku_solution):
            return ThoughtValidity.VALID_INTERMEDIATE
        else:
            return ThoughtValidity.INVALID

"""
Just testing the MyChecker class above:
"""
logger.info("Just testing the MyChecker class above:")

checker = MyChecker()
assert (
    checker.evaluate("", ("3,*,*,2|1,*,3,*|*,1,*,3|4,*,*,1",))
    == ThoughtValidity.VALID_INTERMEDIATE
)
assert (
    checker.evaluate("", ("3,4,1,2|1,2,3,4|2,1,4,3|4,3,2,1",))
    == ThoughtValidity.VALID_FINAL
)
assert (
    checker.evaluate("", ("3,4,1,2|1,2,3,4|2,1,4,3|4,3,*,1",))
    == ThoughtValidity.VALID_INTERMEDIATE
)
assert (
    checker.evaluate("", ("3,4,1,2|1,2,3,4|2,1,4,3|4,*,3,1",))
    == ThoughtValidity.INVALID
)

"""
## Tree of Thought Chain

Initialize and run the ToT chain, with maximum number of interactions `k` set to `30` and the maximum number child thoughts `c` set to `8`.
"""
logger.info("## Tree of Thought Chain")


tot_chain = ToTChain(
    llm=llm, checker=MyChecker(), k=30, c=5, verbose=True, verbose_llm=False
)
tot_chain.run(problem_description=problem_description)

logger.info("\n\n[DONE]", bright=True)