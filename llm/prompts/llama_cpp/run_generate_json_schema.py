import json
import shutil
from pathlib import Path

from jet.file.utils import save_file
from jet.llm.prompt_templates.llama_cpp import (
    generate_json_schema,
)

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    query = "Top villainess anime 2026"

    json_schema = generate_json_schema(query)

    print(json.dumps(json_schema, indent=2, ensure_ascii=False))

    save_file(json_schema, OUTPUT_DIR / "json_schema.json")
