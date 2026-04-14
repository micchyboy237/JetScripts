import json
import shutil
from pathlib import Path

from jet.file.utils import save_file
from jet.llm.prompt_templates.llama_cpp import (
    generate_browser_query_context_json_schema,
)

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    query = "Top villainess anime 2026"
    context = """Scraped web data: From a popular anime blog, it lists the top villainess anime for 2026. 
The list includes titles with brief descriptions, release years, number of episodes, ratings, 
and the name of the main villainess character."""

    json_schema = generate_browser_query_context_json_schema(query, context)
    print("Browser Query + Context JSON Schema:")
    print(json.dumps(json_schema, indent=2, ensure_ascii=False))
    save_file(json_schema, OUTPUT_DIR / "json_schema.json")
