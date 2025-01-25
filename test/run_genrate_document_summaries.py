import os

from tqdm import tqdm
from jet.llm.ollama.base import Ollama, initialize_ollama_settings
from jet.logger import logger
from jet.file import save_file
from llama_index.core.indices.list.base import SummaryIndex
from llama_index.core.readers.file.base import SimpleDirectoryReader
initialize_ollama_settings()

llm = Ollama(model="llama3.2")

data_dir = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/jet-resume/data"
sub_dir = "jet-resume-summary"

documents = SimpleDirectoryReader(
    data_dir, recursive=True).load_data()
generated_dir = os.path.join(
    "generated", os.path.basename(__file__).split('.')[0], sub_dir)
os.makedirs(generated_dir, exist_ok=True)


# Generate summary
doc_titles = [doc.metadata['file_name'] for doc in documents]
generation_tqdm = tqdm(doc_titles, total=len(doc_titles))
for tqdm_idx, file_name in enumerate(generation_tqdm):
    summary_index = SummaryIndex.from_documents(
        documents=[documents[tqdm_idx]],
        show_progress=True,
    )
    summarizer = summary_index.as_query_engine(
        response_mode="tree_summarize", llm=llm
    )

    logger.newline()
    response = summarizer.query(
        f"Summarize the contents of this document.")

    summary = response.response

    output_file = os.path.join(generated_dir, file_name)

    save_file(summary, output_file)

    generation_tqdm.update(1)
