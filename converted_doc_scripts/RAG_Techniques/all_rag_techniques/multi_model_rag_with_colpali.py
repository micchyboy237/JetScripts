from IPython.display import Image
from PIL import Image
from byaldi import RAGMultiModalModel
from jet.logger import CustomLogger
import base64
import google.generativeai as genai
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(script_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.log")
logger = CustomLogger(log_file, overwrite=True)
logger.info(f"Logs: {log_file}")

"""
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/multi_model_rag_with_colpali.ipynb)

### Overview:
This code implements one of the multiple ways of multi-model RAG. This project processes a PDF file, retrieves relevant content using Colpali, and generates answers using a multi-modal RAG system. The process includes document indexing, querying, and summarizing with the Gemini model.

### Key Components:
- **RAGMultiModalModel**: Used for document indexing and retrieval.
- **PDF Processing**: Downloads and processes "Attention is All You Need" paper.
- **Gemini Model**: Used for content generation from retrieved images and queries.
- **Base64 Encoding/Decoding**: Manages image data retrieved during search.

### Diagram:
   <img src="../images/multi_model_rag_with_colpali.svg" alt="Reliable-RAG" width="300">

### Motivation:
To enable efficient querying and content generation from multi-modal documents (PDFs with text and images) in response to natural language queries.

### Method Details:
- Indexing: The PDF is indexed using the `RAGMultiModalModel`, storing both text and image data.
- Querying: Natural language queries retrieve relevant document segments.
- Image Processing: Images from the document are decoded, displayed, and used in conjunction with the Gemini model to generate content.

### Benefits:
- Multi-modal support for both text and images.
- Streamlined retrieval and summarization pipeline.
- Flexible content generation using advanced LLMs (Gemini model).

### Implementation:
- PDF is indexed, and the content is split into text and image segments.
- A query is run against the indexed document to fetch the relevant results.
- Retrieved image data is decoded and passed through the Gemini model for answer generation.

### Summary:
This project integrates document indexing, retrieval, and content generation in a multi-modal setting, enabling efficient queries on complex documents like research papers.

## Setup
"""
logger.info("### Overview:")

# !pip install pillow python-dotenv

# !pip install -q git+https://github.com/huggingface/transformers.git qwen-vl-utils flash-attn optimum auto-gptq bitsandbytes

"""
# Package Installation and Imports

The cell below installs all necessary packages required to run this notebook.
"""
logger.info("# Package Installation and Imports")

# !pip install base64 byaldi os ragmultimodalmodel

"""
# Package Installation

The cell below installs all necessary packages required to run this notebook. If you're running this notebook in a new environment, execute this cell first to ensure all dependencies are installed.
"""
logger.info("# Package Installation")

# !pip install byaldi

os.environ["HF_token"] = 'your-huggingface-api-key' # to download the ColPali model

RAG = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2", verbose=1)

"""
### Download the "Attention is all you need" paper
"""
logger.info("### Download the "Attention is all you need" paper")

# !wget https://arxiv.org/pdf/1706.03762
# !mkdir docs
# !mv 1706.03762 docs/attention_is_all_you_need.pdf

"""
### Indexing
"""
logger.info("### Indexing")

RAG.index(
    input_path="./docs/attention_is_all_you_need.pdf",
    index_name="attention_is_all_you_need",
    store_collection_with_index=True, # set this to false if you don't want to store the base64 representation
    overwrite=True
)

"""
### Query time
"""
logger.info("### Query time")

query = "What is the BLEU score of the Transformer (base model)?"

results = RAG.search(query, k=1)

"""
### Actual image data
"""
logger.info("### Actual image data")

image_bytes = base64.b64decode(results[0].base64)

filename = 'image.jpg'  # I assume you have a JPG file
with open(filename, 'wb') as f:
  f.write(image_bytes)


display(Image(filename))

"""
## Test using gemini-1.5-flash
"""
logger.info("## Test using gemini-1.5-flash")


genai.configure(api_key='your-api-key')
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

image = Image.open(filename)

response = model.generate_content([image, query])
logger.debug(response.text)

logger.info("\n\n[DONE]", bright=True)