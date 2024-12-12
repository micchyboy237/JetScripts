import os
from tqdm import tqdm
from aim import Run, Text
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_core.documents import Document

docs_dir = "/Users/jethroestrada/Desktop/External_Projects/AI/chatbot/open-webui/backend/crewAI/docs"
repo_dir = "./"


# Initialize the GenericLoader
loader = GenericLoader.from_filesystem(
    docs_dir,
    glob="**/*",
    suffixes=[".md", ".mdx"],
    parser=LanguageParser(),
)

# Load the documents
docs = loader.load()

# Initialize a new run
run = Run(repo=repo_dir)

# Wrap the loop with tqdm for progress tracking
for step, doc in tqdm(enumerate(docs), total=len(docs), desc="Processing documents"):
    item: Document = doc
    file_path = item.metadata['source']
    rel_path = os.path.relpath(file_path, start=docs_dir)
    aim_text = Text(item.page_content)
    run.track(aim_text, name='text', step=step, context={
        "from": "CrewAI Docs",
        "rel_path": rel_path,
    })
