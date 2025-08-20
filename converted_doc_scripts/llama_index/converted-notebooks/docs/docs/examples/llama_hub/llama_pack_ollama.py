from jet.logger import CustomLogger
from jet.models.config import MODELS_CACHE_DIR
from llama_index.core import SimpleDirectoryReader
from llama_index.core.llama_pack import download_llama_pack
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
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
# Ollama Llama Pack Example

### Setup Data
"""
logger.info("# Ollama Llama Pack Example")

# !wget "https://www.dropbox.com/s/f6bmb19xdg0xedm/paul_graham_essay.txt?dl=1" -O paul_graham_essay.txt


reader = SimpleDirectoryReader(input_files=["paul_graham_essay.txt"])
documents = reader.load_data()

"""
### Start Ollama

Make sure you run `ollama run llama2` in a terminal.
"""
logger.info("### Start Ollama")



"""
### Download and Initialize Pack

We use `download_llama_pack` to download the pack class, and then we initialize it with documents.

Every pack will have different initialization parameters. You can find more about the initialization parameters for each pack through its [README](https://github.com/logan-markewich/llama-hub/tree/main/llama_hub/llama_packs/voyage_query_engine) (also on LlamaHub).

**NOTE**: You must also specify an output directory. In this case the pack is downloaded to `voyage_pack`. This allows you to customize and make changes to the file, and import it later!
"""
logger.info("### Download and Initialize Pack")


OllamaQueryEnginePack = download_llama_pack(
    "OllamaQueryEnginePack", "./ollama_pack"
)

ollama_pack = OllamaQueryEnginePack(model="llama2", documents=documents)

response = ollama_pack.run("What did the author do growing up?")

logger.debug(str(response))

logger.info("\n\n[DONE]", bright=True)