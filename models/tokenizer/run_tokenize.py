from jet._token.token_utils import tokenize
from jet.file.utils import save_file
import os
import shutil

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    model = "qwen3-instruct-2507:4b"
    text = "Sample text to tokenize"
    tokens = tokenize(text, model)
    save_file(tokens, f"{OUTPUT_DIR}/encoded.json")
    
    batch_tokens = tokenize([text], model)
    save_file(batch_tokens, f"{OUTPUT_DIR}/encoded_batch.json")
