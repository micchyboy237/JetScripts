from jet.llm.mlx.base import MLX
from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core.llama_pack import download_llama_pack
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.packs.self_discover import SelfDiscoverPack
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

model_name = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(
    model_name=model_name,
    cache_folder=MODELS_CACHE_DIR,
)


"""
# Self Discover Pack

<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/llama-index-packs/llama-index-packs-self-discover/examples/self_discover.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

This LlamaPack implements [Self-Discover: Large Language Models Self-Compose Reasoning Structures](https://arxiv.org/abs/2402.03620) paper.

It has two stages for the given task:

1. STAGE-1:

    a. SELECT: Selects subset of reasoning Modules.

    b. ADAPT: Adapts selected reasoning modules to the task.

    c. IMPLEMENT: It gives reasoning structure for the task.
    
2. STAGE-2: Uses the generated reasoning structure for the task to generate an answer.


The implementation is inspired from the [codebase](https://github.com/catid/self-discover)

### Setup
"""
logger.info("# Self Discover Pack")

# import nest_asyncio

# nest_asyncio.apply()


# os.environ["OPENAI_API_KEY"] = "<Your MLX API Key>"

llm = MLX(model="qwen3-1.7b-4bit", log_dir=f"{OUTPUT_DIR}/chats")

"""
### Load / Download Pack

There are two ways to use the LlamaPack.

1. Do `download_llama_pack` to load the Self-Discover LlamaPack.
2. Directly use `SelfDiscoverPack`

#### Using `download_llama_pack`
"""
logger.info("### Load / Download Pack")


SelfDiscoverPack = download_llama_pack("SelfDiscoverPack", "./self_discover_pack")

self_discover_pack = SelfDiscoverPack(verbose=True, llm=llm)

"""
#### Directly use `SelfDiscoverPack`
"""
logger.info("#### Directly use `SelfDiscoverPack`")


self_discover_pack = SelfDiscoverPack(verbose=True, llm=llm)

"""
### Test out on some tasks

#### Task

Michael has 15 oranges. He gives 4 oranges to his brother and trades 3 oranges for 6 apples with his neighbor. Later in the day, he realizes some of his oranges are spoiled, so he discards 2 of them. Then, Michael goes to the market and buys 12 more oranges and 5 more apples. If Michael decides to give 2 apples to his friend, how many oranges and apples does Michael have now?
"""
logger.info("### Test out on some tasks")

task = "Michael has 15 oranges. He gives 4 oranges to his brother and trades 3 oranges for 6 apples with his neighbor. Later in the day, he realizes some of his oranges are spoiled, so he discards 2 of them. Then, Michael goes to the market and buys 12 more oranges and 5 more apples. If Michael decides to give 2 apples to his friend, how many oranges and apples does Michael have now?"
output = self_discover_pack.run(task)

logger.debug(output)

"""
#### Task

Tom needs to buy ingredients for a spaghetti dinner he's planning for himself and a friend. He already has pasta at home but needs to buy tomato sauce and ground beef. At the store, tomato sauce costs $3 per jar and ground beef costs $5 per pound. Tom buys 2 jars of tomato sauce and 2 pounds of ground beef. He also picks up a loaf of bread for $2.50. If Tom pays with a $50 bill, how much change should he receive after purchasing these items?
"""
logger.info("#### Task")

task = "Tom needs to buy ingredients for a spaghetti dinner he's planning for himself and a friend. He already has pasta at home but needs to buy tomato sauce and ground beef. At the store, tomato sauce costs $3 per jar and ground beef costs $5 per pound. Tom buys 2 jars of tomato sauce and 2 pounds of ground beef. He also picks up a loaf of bread for $2.50. If Tom pays with a $50 bill, how much change should he receive after purchasing these items?"
output = self_discover_pack.run(task)

logger.debug(output)

logger.info("\n\n[DONE]", bright=True)