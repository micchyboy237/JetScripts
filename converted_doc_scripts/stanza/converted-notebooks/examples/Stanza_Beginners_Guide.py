from jet.logger import logger
import os
import shutil
import stanza


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

"""
# Welcome to Stanza!

![Latest Version](https://img.shields.io/pypi/v/stanza.svg?colorB=bc4545)
![Python Versions](https://img.shields.io/pypi/pyversions/stanza.svg?colorB=bc4545)

Stanza is a Python NLP toolkit that supports 60+ human languages. It is built with highly accurate neural network components that enable efficient training and evaluation with your own annotated data, and offers pretrained models on 100 treebanks. Additionally, Stanza provides a stable, officially maintained Python interface to Java Stanford CoreNLP Toolkit.

In this tutorial, we will demonstrate how to set up Stanza and annotate text with its native neural network NLP models. For the use of the Python CoreNLP interface, please see other tutorials.

## 1. Installing Stanza

Note that Stanza only supports Python 3.6 and above. Installing and importing Stanza are as simple as running the following commands:
"""
logger.info("# Welcome to Stanza!")

# !pip install stanza


"""
### More Information

For common troubleshooting, please visit our [troubleshooting page](https://stanfordnlp.github.io/stanfordnlp/installation_usage.html#troubleshooting).

## 2. Downloading Models

You can download models with the `stanza.download` command. The language can be specified with either a full language name (e.g., "english"), or a short code (e.g., "en"). 

By default, models will be saved to your `~/stanza_resources` directory. If you want to specify your own path to save the model files, you can pass a `dir=your_path` argument.
"""
logger.info("### More Information")

# logger.debug("Downloading English model...")
# stanza.download('en')

# logger.debug("Downloading Chinese model...")
# stanza.download('zh', verbose=True)

"""
### More Information

Pretrained models are provided for 60+ different languages. For all languages, available models and the corresponding short language codes, please check out the [models page](https://stanfordnlp.github.io/stanza/models.html).

## 3. Processing Text

### Constructing Pipeline

To process a piece of text, you'll need to first construct a `Pipeline` with different `Processor` units. The pipeline is language-specific, so again you'll need to first specify the language (see examples).

- By default, the pipeline will include all processors, including tokenization, multi-word token expansion, part-of-speech tagging, lemmatization, dependency parsing and named entity recognition (for supported languages). However, you can always specify what processors you want to include with the `processors` argument.

- Stanza's pipeline is CUDA-aware, meaning that a CUDA-device will be used whenever it is available, otherwise CPUs will be used when a GPU is not found. You can force the pipeline to use CPU regardless by setting `use_gpu=False`.

- Again, you can suppress all printed messages by setting `verbose=False`.
"""
logger.info("### More Information")

logger.debug("Building an English pipeline...")
en_nlp = stanza.Pipeline('en')

logger.debug("Building a Chinese pipeline...")
zh_nlp = stanza.Pipeline('zh', processors='tokenize,lemma,pos,depparse', verbose=True, use_gpu=False)

"""
### Annotating Text

After a pipeline is successfully constructed, you can get annotations of a piece of text simply by passing the string into the pipeline object. The pipeline will return a `Document` object, which can be used to access detailed annotations from. For example:
"""
logger.info("### Annotating Text")

en_doc = en_nlp("Barack Obama was born in Hawaii.  He was elected president in 2008.")
logger.debug(type(en_doc))

zh_doc = zh_nlp("达沃斯世界经济论坛是每年全球政商界领袖聚在一起的年度盛事。")
logger.debug(type(zh_doc))

"""
### More Information

For more information on how to construct a pipeline and information on different processors, please visit our [pipeline page](https://stanfordnlp.github.io/stanfordnlp/pipeline.html).

## 4. Accessing Annotations

Annotations can be accessed from the returned `Document` object. 

A `Document` contains a list of `Sentence`s, and a `Sentence` contains a list of `Token`s and `Word`s. For the most part `Token`s and `Word`s overlap, but some tokens can be divided into mutiple words, for instance the French token `aux` is divided into the words `à` and `les`, while in English a word and a token are equivalent. Note that dependency parses are derived over `Word`s.

Additionally, a `Span` object is used to represent annotations that are part of a document, such as named entity mentions.


The following example iterate over all English sentences and words, and print the word information one by one:
"""
logger.info("### More Information")

for i, sent in enumerate(en_doc.sentences):
    logger.debug("[Sentence {}]".format(i+1))
    for word in sent.words:
        logger.debug("{:12s}\t{:12s}\t{:6s}\t{:d}\t{:12s}".format(\
              word.text, word.lemma, word.pos, word.head, word.deprel))
    logger.debug("")

"""
The following example iterate over all extracted named entity mentions and print out their character spans and types.
"""
logger.info("The following example iterate over all extracted named entity mentions and print out their character spans and types.")

logger.debug("Mention text\tType\tStart-End")
for ent in en_doc.ents:
    logger.debug("{}\t{}\t{}-{}".format(ent.text, ent.type, ent.start_char, ent.end_char))

"""
And similarly for the Chinese text:
"""
logger.info("And similarly for the Chinese text:")

for i, sent in enumerate(zh_doc.sentences):
    logger.debug("[Sentence {}]".format(i+1))
    for word in sent.words:
        logger.debug("{:12s}\t{:12s}\t{:6s}\t{:d}\t{:12s}".format(\
              word.text, word.lemma, word.pos, word.head, word.deprel))
    logger.debug("")

"""
Alternatively, you can directly print a `Word` object to view all its annotations as a Python dict:
"""
logger.info("Alternatively, you can directly print a `Word` object to view all its annotations as a Python dict:")

word = en_doc.sentences[0].words[0]
logger.debug(word)

"""
### More Information

For all information on different data objects, please visit our [data objects page](https://stanfordnlp.github.io/stanza/data_objects.html).

## 5. Resources

Apart from this interactive tutorial, we also provide tutorials on our website that cover a variety of use cases such as how to use different model "packages" for a language, how to use spaCy as a tokenizer, how to process pretokenized text without running the tokenizer, etc. For these tutorials please visit [our Tutorials page](https://stanfordnlp.github.io/stanza/tutorials.html).

Other resources that you may find helpful include:

- [Stanza Homepage](https://stanfordnlp.github.io/stanza/index.html)
- [FAQs](https://stanfordnlp.github.io/stanza/faq.html)
- [GitHub Repo](https://github.com/stanfordnlp/stanza)
- [Reporting Issues](https://github.com/stanfordnlp/stanza/issues)
- [Stanza System Description Paper](http://arxiv.org/abs/2003.07082)
"""
logger.info("### More Information")

logger.info("\n\n[DONE]", bright=True)