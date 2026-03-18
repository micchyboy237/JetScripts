import os
import shutil
from pathlib import Path

from datasets import load_dataset
from jet.adapters.llama_cpp.types import LLAMACPP_EMBED_KEYS
from jet.adapters.llama_cpp.vector_search import VectorSearch
from jet.file.utils import save_file
from jet.logger.config import colorize_log

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

query = "text translation between Japanese and English"

ds = load_dataset("TFMC/imatrix-dataset-for-japanese-llm", split="train")

docs = [o["text"] for o in ds.to_list() if o["text"]]

save_file(
    {
        "count": len(docs),
        "texts": docs,
    },
    OUTPUT_DIR / "docs.json",
)

model: LLAMACPP_EMBED_KEYS = os.getenv("LLAMA_CPP_EMBED_MODEL")
search_engine = VectorSearch(model)

results = search_engine.search(query, docs)

print(f"\nQuery: {query}")
print("Top matches:")
for result in results[:10]:
    print(
        f"\n{colorize_log(f'{result["rank"]}.', 'ORANGE')} (Score: "
        f"{colorize_log(f'{result["score"]:.3f}', 'SUCCESS')})"
    )
    print(f"{result['text']}")

save_file(
    {
        "count": len(results),
        "results": results,
    },
    OUTPUT_DIR / "search_results.json",
)
