from flair.embeddings import TransformerDocumentEmbeddings
from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings
from jet.logger import logger
from keybert import KeyBERT
from keybert.backend import BaseEmbedder
from keybert.backend import Model2VecBackend
from model2vec import StaticModel
from sentence_transformers import SentenceTransformer
from thinc.api import set_gpu_allocator, require_gpu
from transformers.pipelines import pipeline
import gensim.downloader as api
import os
import shutil
import spacy
import tensorflow_hub


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

"""
# Embedding Models
In this tutorial we will be going through the embedding models that can be used in KeyBERT.
Having the option to choose embedding models allow you to leverage pre-trained embeddings that suit your use-case.

### **Sentence Transformers**
You can select any model from sentence-transformers [here](https://www.sbert.net/docs/pretrained_models.html)
and pass it through KeyBERT with `model`:
"""
logger.info("# Embedding Models")

kw_model = KeyBERT(model="all-MiniLM-L6-v2")

"""
Or select a SentenceTransformer model with your own parameters:
"""
logger.info("Or select a SentenceTransformer model with your own parameters:")


sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
kw_model = KeyBERT(model=sentence_model)

"""
### **Model2Vec**

For blazingly fast embedding models, [Model2Vec](https://github.com/MinishLab/model2vec) is an incredible framework. To use it KeyBERT, you only need to pass their `StaticModel`:
"""
logger.info("### **Model2Vec**")


embedding_model = StaticModel.from_pretrained("minishlab/potion-base-8M")
kw_model = KeyBERT(embedding_model)

"""
If you want to distill a sentence-transformers model with the vocabulary of the documents,
run the following:
"""
logger.info("If you want to distill a sentence-transformers model with the vocabulary of the documents,")


embedding_model = Model2VecBackend("sentence-transformers/all-MiniLM-L6-v2", distill=True)

"""
Note that this is especially helpful if you have a very large dataset (I wouldn't recommend it with small datasets).

!!! Tip
    If you also want to have a light-weight installation without (sentence-)transformers, you can install KeyBERT as follows:
    `pip install keybert --no-deps scikit-learn model2vec`
    This will make the installation much smaller and the import much quicker.

### 🤗 **Hugging Face Transformers**
To use a Hugging Face transformers model, load in a pipeline and point
to any model found on their model hub (https://huggingface.co/models):
"""
logger.info("### 🤗 **Hugging Face Transformers**")


hf_model = pipeline("feature-extraction", model="distilbert-base-cased")
kw_model = KeyBERT(model=hf_model)

"""
!!! tip "Tip!"
    These transformers also work quite well using `sentence-transformers` which has a number of
    optimizations tricks that make using it a bit faster.

### **Flair**
[Flair](https://github.com/flairNLP/flair) allows you to choose almost any embedding model that
is publicly available. Flair can be used as follows:
"""
logger.info("### **Flair**")


roberta = TransformerDocumentEmbeddings('roberta-base')
kw_model = KeyBERT(model=roberta)

"""
You can select any 🤗 transformers model [here](https://huggingface.co/models).

Moreover, you can also use Flair to use word embeddings and pool them to create document embeddings.
Under the hood, Flair simply averages all word embeddings in a document. Then, we can easily
pass it to KeyBERT in order to use those word embeddings as document embeddings:
"""
logger.info("You can select any 🤗 transformers model [here](https://huggingface.co/models).")


glove_embedding = WordEmbeddings('crawl')
document_glove_embeddings = DocumentPoolEmbeddings([glove_embedding])

kw_model = KeyBERT(model=document_glove_embeddings)

"""
### **Spacy**
[Spacy](https://github.com/explosion/spaCy) is an amazing framework for processing text. There are
many models available across many languages for modeling text.

To use Spacy's non-transformer models in KeyBERT:
"""
logger.info("### **Spacy**")


nlp = spacy.load("en_core_web_md", exclude=['tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])

kw_model = KeyBERT(model=nlp)

"""
Using spacy-transformer models:
"""
logger.info("Using spacy-transformer models:")


spacy.prefer_gpu()
nlp = spacy.load("en_core_web_trf", exclude=['tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])

kw_model = KeyBERT(model=nlp)

"""
If you run into memory issues with spacy-transformer models, try:
"""
logger.info("If you run into memory issues with spacy-transformer models, try:")


nlp = spacy.load("en_core_web_trf", exclude=['tagger', 'parser', 'ner', 'attribute_ruler', 'lemmatizer'])
set_gpu_allocator("pytorch")
require_gpu(0)

kw_model = KeyBERT(model=nlp)

"""
### **Universal Sentence Encoder (USE)**
The Universal Sentence Encoder encodes text into high dimensional vectors that are used here
for embedding the documents. The model is trained and optimized for greater-than-word length text,
such as sentences, phrases or short paragraphs.

Using USE in KeyBERT is rather straightforward:
"""
logger.info("### **Universal Sentence Encoder (USE)**")

embedding_model = tensorflow_hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
kw_model = KeyBERT(model=embedding_model)

"""
### **Gensim**
For Gensim, KeyBERT supports its `gensim.downloader` module. Here, we can download any model word embedding model
to be used in KeyBERT. Note that Gensim is primarily used for Word Embedding models. This works typically
best for short documents since the word embeddings are pooled.
"""
logger.info("### **Gensim**")

ft = api.load('fasttext-wiki-news-subwords-300')
kw_model = KeyBERT(model=ft)

"""
### **Custom Backend**
If your backend or model cannot be found in the ones currently available, you can use the `keybert.backend.BaseEmbedder` class to
create your own backend. Below, you will find an example of creating a SentenceTransformer backend for KeyBERT:
"""
logger.info("### **Custom Backend**")


class CustomEmbedder(BaseEmbedder):
    def __init__(self, embedding_model):
        super().__init__()
        self.embedding_model = embedding_model

    def embed(self, documents, verbose=False):
        embeddings = self.embedding_model.encode(documents, show_progress_bar=verbose)
        return embeddings

distilbert = SentenceTransformer("paraphrase-MiniLM-L6-v2")
custom_embedder = CustomEmbedder(embedding_model=distilbert)

kw_model = KeyBERT(model=custom_embedder)

logger.info("\n\n[DONE]", bright=True)