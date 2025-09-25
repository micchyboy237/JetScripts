from jet.logger import logger
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
## Assignment: Diophantine Equations

> This assignment is part of [AI for Beginners Curriculum](http://github.com/microsoft/ai-for-beginners) and is inspired by [this post](https://habr.com/post/128704/).

Your goal is to solve so-called **Diophantine equation** - an equation with integer roots and integer coefficients. For example, consider the following equation:

$$a+2b+3c+4d=30$$

You need to find integer roots $a$,$b$,$c$,$d\in\mathbb{N}$ that satisfy this equation.

Hints:
1. You can consider roots to be in the interval [0;30]
1. As a gene, consider using the list of root values


"""
logger.info("## Assignment: Diophantine Equations")

logger.info("\n\n[DONE]", bright=True)