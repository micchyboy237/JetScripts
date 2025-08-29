from jet.logger import CustomLogger
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser import SimpleFileNodeParser
from llama_index.readers.file import FlatReader
from pathlib import Path
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
<a href="https://colab.research.google.com/github/run-llama/llama_index/blob/main/docs/docs/examples/node_postprocessor/FileNodeProcessors.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# File Based Node Parsers

The `SimpleFileNodeParser` and `FlatReader` are designed to allow opening a variety of file types and automatically selecting the best `NodeParser` to process the files. The `FlatReader` loads the file in a raw text format and attaches the file information to the metadata, then the `SimpleFileNodeParser` maps file types to node parsers in `node_parser/file`, selecting the best node parser for the job.

The `SimpleFileNodeParser` does not perform token based chunking of the text, and is intended to be used in combination with a token node parser.

Let's look at an example of using the `FlatReader` and `SimpleFileNodeParser` to load content. For the README file I will be using the LlamaIndex README and the HTML file is the Stack Overflow landing page, however any README and HTML file will work.

If you're opening this Notebook on colab, you will probably need to install LlamaIndex 🦙.
"""
logger.info("# File Based Node Parsers")

# %pip install llama-index-readers-file

# !pip install llama-index


reader = FlatReader()
html_file = reader.load_data(Path("./stack-overflow.html"))
md_file = reader.load_data(Path("./README.md"))
logger.debug(html_file[0].metadata)
logger.debug(html_file[0])
logger.debug("----")
logger.debug(md_file[0].metadata)
logger.debug(md_file[0])

"""
## Parsing the files

The flat reader has simple loaded the content of the files into Document objects for further processing. We can see that the file information is retained in the metadata. Let's pass the documents to the node parser to see the parsing.
"""
logger.info("## Parsing the files")

parser = SimpleFileNodeParser()
md_nodes = parser.get_nodes_from_documents(md_file)
html_nodes = parser.get_nodes_from_documents(html_file)
logger.debug(md_nodes[0].metadata)
logger.debug(md_nodes[0].text)
logger.debug(md_nodes[1].metadata)
logger.debug(md_nodes[1].text)
logger.debug("----")
logger.debug(html_nodes[0].metadata)
logger.debug(html_nodes[0].text)

"""
## Furter processing of files

We can see that the Markdown and HTML files have been split into chunks based on the structure of the document. The markdown node parser splits on any headers and attaches the hierarchy of headers into metadata. The HTML node parser extracted text from common text elements to simplifiy the HTML file, and combines neighbouring nodes of the same element. Compared to working with raw HTML, this is alreadly a big improvement in terms of retrieving meaningful text content.

Because these files were only split according to the structure of the file, we can apply further processing with a text splitter to prepare the content into nodes of limited token length.
"""
logger.info("## Furter processing of files")


splitting_parser = SentenceSplitter(chunk_size=200, chunk_overlap=0)

html_chunked_nodes = splitting_parser(html_nodes)
md_chunked_nodes = splitting_parser(md_nodes)
logger.debug(f"\n\nHTML parsed nodes: {len(html_nodes)}")
logger.debug(html_nodes[0].text)

logger.debug(f"\n\nHTML chunked nodes: {len(html_chunked_nodes)}")
logger.debug(html_chunked_nodes[0].text)

logger.debug(f"\n\nMD parsed nodes: {len(md_nodes)}")
logger.debug(md_nodes[0].text)

logger.debug(f"\n\nMD chunked nodes: {len(md_chunked_nodes)}")
logger.debug(md_chunked_nodes[0].text)

"""
## Summary

We can see that the files have been further processed within the splits created by `SimpleFileNodeParser`, and are now ready to be ingested by an index or vector store. The code cell below shows just the chaining of the parsers to go from raw file to chunked nodes:
"""
logger.info("## Summary")


pipeline = IngestionPipeline(
    documents=reader.load_data(Path("./README.md")),
    transformations=[
        SimpleFileNodeParser(),
        SentenceSplitter(chunk_size=200, chunk_overlap=0),
    ],
)

md_chunked_nodes = pipeline.run()
logger.debug(md_chunked_nodes)

logger.info("\n\n[DONE]", bright=True)