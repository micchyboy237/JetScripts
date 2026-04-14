import shutil
from pathlib import Path

from jet.file.utils import save_file
from jet.llm.prompt_templates.llama_cpp import (
    generate_field_descriptions,
)

OUTPUT_DIR = Path(__file__).parent / "generated" / Path(__file__).stem
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    query = "Top villainess anime 2026"
    field_descriptions = generate_field_descriptions(query)
    print("Field Descriptions:")
    print(field_descriptions)
    save_file(field_descriptions, OUTPUT_DIR / "field_descriptions.txt")
