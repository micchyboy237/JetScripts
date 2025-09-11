from jet.adapters.langchain.chat_ollama import Ollama
from jet.logger import logger
from langchain_experimental.llm_symbolic_math.base import LLMSymbolicMathChain
import os
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
# LLM Symbolic Math 
This notebook showcases using LLMs and Python to Solve Algebraic Equations. Under the hood is makes use of [SymPy](https://www.sympy.org/en/index.html).
"""
logger.info("# LLM Symbolic Math")


llm = Ollama(temperature=0)
llm_symbolic_math = LLMSymbolicMathChain.from_llm(llm)

"""
## Integrals and derivates
"""
logger.info("## Integrals and derivates")

llm_symbolic_math.invoke("What is the derivative of sin(x)*exp(x) with respect to x?")

llm_symbolic_math.invoke(
    "What is the integral of exp(x)*sin(x) + exp(x)*cos(x) with respect to x?"
)

"""
## Solve linear and differential equations
"""
logger.info("## Solve linear and differential equations")

llm_symbolic_math.invoke('Solve the differential equation y" - y = e^t')

llm_symbolic_math.invoke("What are the solutions to this equation y^3 + 1/3y?")

llm_symbolic_math.invoke("x = y + 5, y = z - 3, z = x * y. Solve for x, y, z")

logger.info("\n\n[DONE]", bright=True)