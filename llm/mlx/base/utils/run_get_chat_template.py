import os
import shutil

from jet.llm.mlx.mlx_utils import get_chat_template
from jet.models.model_types import LLMModelType
from jet.file.utils import save_file
from jet.models.utils import resolve_model_key
from jet.utils.text import format_sub_dir

OUTPUT_DIR = os.path.join(os.path.dirname(
    __file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)


def main(model: LLMModelType):
    chat_template = get_chat_template(model)
    filename = format_sub_dir(resolve_model_key(model))
    save_file(chat_template, f"{OUTPUT_DIR}/{filename}.jinja")


if __name__ == "__main__":
    main("mlx-community/Llama-3.2-3B-Instruct-4bit")
    main("mlx-community/Mistral-7B-Instruct-v0.3-4bit")
