import shutil
from pathlib import Path

from jet.file.utils import save_file
from jet.libs.bertopic.examples.mock import load_sample_jobs_ai_llm_python
from jet.transformers.object import make_serializable
from jet.wordnet.analyzers.analyze_ner import analyze_ner

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    texts = load_sample_jobs_ai_llm_python()

    result = analyze_ner(texts)

    for key, value in result.items():
        save_file(make_serializable(value), OUTPUT_DIR / f"{key}.json")
