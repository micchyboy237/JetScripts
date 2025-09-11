from jet.adapters.langchain.chat_ollama import ChatOllama
from jet.logger import logger
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import AmazonTextractPDFLoader
from textractor.data.text_linearization_config import TextLinearizationConfig
import boto3
import os
import shutil


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

PERSIST_DIR = f"{OUTPUT_DIR}/chroma"
os.makedirs(PERSIST_DIR, exist_ok=True)

"""
# Amazon Textract 

>[Amazon Textract](https://docs.aws.amazon.com/managedservices/latest/userguide/textract.html) is a machine learning (ML) service that automatically extracts text, handwriting, and data from scanned documents.
>
>It goes beyond simple optical character recognition (OCR) to identify, understand, and extract data from forms and tables. Today, many companies manually extract data from scanned documents such as PDFs, images, tables, and forms, or through simple OCR software that requires manual configuration (which often must be updated when the form changes). To overcome these manual and expensive processes, `Textract` uses ML to read and process any type of document, accurately extracting text, handwriting, tables, and other data with no manual effort. 

`Textract` supports `JPEG`, `PNG`, `PDF`, and `TIFF` file formats; more information is available in [the documentation](https://docs.aws.amazon.com/textract/latest/dg/limits-document.html).

The following examples demonstrate the use of `Amazon Textract` in combination with LangChain as a DocumentLoader.
"""
logger.info("# Amazon Textract")

# %pip install --upgrade --quiet  boto3 langchain-ollama tiktoken python-dotenv

# %pip install --upgrade --quiet  "amazon-textract-caller>=0.2.0"

"""
## Example 1: Loading from a local file

The first example uses a local file, which internally will be sent to Amazon Textract sync API [DetectDocumentText](https://docs.aws.amazon.com/textract/latest/dg/API_DetectDocumentText.html). 

Local files or URL endpoints like HTTP:// are limited to one page documents for Textract.
Multi-page documents have to reside on S3. This sample file is a jpeg.
"""
logger.info("## Example 1: Loading from a local file")


loader = AmazonTextractPDFLoader(
    "example_data/alejandro_rosalez_sample-small.jpeg")
documents = loader.load()

"""
Output from the file
"""
logger.info("Output from the file")

documents

"""
## Example 2: Loading from a URL
The next example loads a file from an HTTPS endpoint. 
It has to be single page, as Amazon Textract requires all multi-page documents to be stored on S3.
"""
logger.info("## Example 2: Loading from a URL")


loader = AmazonTextractPDFLoader(
    "https://amazon-textract-public-content.s3.us-east-2.amazonaws.com/langchain/alejandro_rosalez_sample_1.jpg"
)
documents = loader.load()

documents

"""
## Example 3: Loading multi-page PDF documents

Processing a multi-page document requires the document to be on S3. The sample document resides in a bucket in us-east-2 and Textract needs to be called in that same region to be successful, so we set the region_name on the client and pass that in to the loader to ensure Textract is called from us-east-2. You could also to have your notebook running in us-east-2, setting the AWS_DEFAULT_REGION set to us-east-2 or when running in a different environment, pass in a boto3 Textract client with that region name like in the cell below.
"""
logger.info("## Example 3: Loading multi-page PDF documents")


textract_client = boto3.client("textract", region_name="us-east-2")

file_path = "s3://amazon-textract-public-content/langchain/layout-parser-paper.pdf"
loader = AmazonTextractPDFLoader(file_path, client=textract_client)
documents = loader.load()

"""
Now getting the number of pages to validate the response (printing out the full response would be quite long...). We expect 16 pages.
"""
logger.info("Now getting the number of pages to validate the response (printing out the full response would be quite long...). We expect 16 pages.")

len(documents)

"""
## Example 4: Customizing the output format

When Amazon Textract processes a PDF, it extracts all text, including elements like headers, footers, and page numbers. This extra information can be "noisy" and reduce the effectiveness of the output.

The process of converting a document's 2D layout into a clean, one-dimensional string of text is called linearization.

The AmazonTextractPDFLoader gives you precise control over this process with the `linearization_config` parameter. You can use it to specify which elements to exclude from the final output.

The following example shows how to hide headers, footers, and figures, resulting in a much cleaner text block, for more advanced use cases see this [AWS blog post](https://aws.amazon.com/blogs/machine-learning/amazon-textracts-new-layout-feature-introduces-efficiencies-in-general-purpose-and-generative-ai-document-processing-tasks/).
"""
logger.info("## Example 4: Customizing the output format")


loader = AmazonTextractPDFLoader(
    "s3://amazon-textract-public-content/langchain/layout-parser-paper.pdf",
    linearization_config=TextLinearizationConfig(
        hide_header_layout=True,
        hide_footer_layout=True,
        hide_figure_layout=True,
    ),
)
documents = loader.load()

"""
## Using the AmazonTextractPDFLoader in a LangChain chain (e.g. Ollama)

The AmazonTextractPDFLoader can be used in a chain the same way the other loaders are used.
Textract itself does have a [Query feature](https://docs.aws.amazon.com/textract/latest/dg/API_Query.html), which offers similar functionality to the QA chain in this example, which is worth checking out as well.
"""
logger.info(
    "## Using the AmazonTextractPDFLoader in a LangChain chain (e.g. Ollama)")


# os.environ["OPENAI_API_KEY"] = "your-Ollama-API-key"


chain = load_qa_chain(llm=Ollama(), chain_type="map_reduce")
query = ["Who are the authors?"]

chain.run(input_documents=documents, question=query)

logger.info("\n\n[DONE]", bright=True)
