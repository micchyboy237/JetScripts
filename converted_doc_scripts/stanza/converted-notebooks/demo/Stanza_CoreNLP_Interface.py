from jet.logger import logger
from stanza.server import CoreNLPClient
import os
import shutil
import stanza
import time; time.sleep(10)


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file = os.path.join(OUTPUT_DIR, "main.log")
logger.basicConfig(filename=log_file)
logger.info(f"Logs: {log_file}")

"""
# Stanza: A Tutorial on the Python CoreNLP Interface

![Latest Version](https://img.shields.io/pypi/v/stanza.svg?colorB=bc4545)
![Python Versions](https://img.shields.io/pypi/pyversions/stanza.svg?colorB=bc4545)

While the Stanza library implements accurate neural network modules for basic functionalities such as part-of-speech tagging and dependency parsing, the [Stanford CoreNLP Java library](https://stanfordnlp.github.io/CoreNLP/) has been developed for years and offers more complementary features such as coreference resolution and relation extraction. To unlock these features, the Stanza library also offers an officially maintained Python interface to the CoreNLP Java library. This interface allows you to get NLP anntotations from CoreNLP by writing native Python code.


This tutorial walks you through the installation, setup and basic usage of this Python CoreNLP interface. If you want to learn how to use the neural network components in Stanza, please refer to other tutorials.

## 1. Installation

Before the installation starts, please make sure that you have Python 3 and Java installed on your computer. Since Colab already has them installed, we'll skip this procedure in this notebook.

### Installing Stanza

Installing and importing Stanza are as simple as running the following commands:
"""
logger.info("# Stanza: A Tutorial on the Python CoreNLP Interface")

# !pip install stanza


"""
### Setting up Stanford CoreNLP

In order for the interface to work, the Stanford CoreNLP library has to be installed and a `CORENLP_HOME` environment variable has to be pointed to the installation location.

Here we are going to show you how to download and install the CoreNLP library on your machine, with Stanza's installation command:
"""
logger.info("### Setting up Stanford CoreNLP")

corenlp_dir = './corenlp'
stanza.install_corenlp(dir=corenlp_dir)

os.environ["CORENLP_HOME"] = corenlp_dir

"""
That's all for the installation! 🎉  We can now double check if the installation is successful by listing files in the CoreNLP directory. You should be able to see a number of `.jar` files by running the following command:
"""
logger.info("That's all for the installation! 🎉  We can now double check if the installation is successful by listing files in the CoreNLP directory. You should be able to see a number of `.jar` files by running the following command:")

# !ls $CORENLP_HOME

"""
**Note 1**:
If you are want to use the interface in a terminal (instead of a Colab notebook), you can properly set the `CORENLP_HOME` environment variable with:

```bash
export CORENLP_HOME=path_to_corenlp_dir
```

Here we instead set this variable with the Python `os` library, simply because `export` command is not well-supported in Colab notebook.


**Note 2**:
The `stanza.install_corenlp()` function is only available since Stanza v1.1.1. If you are using an earlier version of Stanza, please check out our [manual installation page](https://stanfordnlp.github.io/stanza/client_setup.html#manual-installation) for how to install CoreNLP on your computer.

**Note 3**:
Besides the installation function, we also provide a `stanza.download_corenlp_models()` function to help you download additional CoreNLP models for different languages that are not shipped with the default installation. Check out our [automatic installation website page](https://stanfordnlp.github.io/stanza/client_setup.html#automated-installation) for more information on how to use it.

## 2. Annotating Text with CoreNLP Interface

### Constructing CoreNLPClient

At a high level, the CoreNLP Python interface works by first starting a background Java CoreNLP server process, and then initializing a client instance in Python which can pass the text to the background server process, and accept the returned annotation results.

We wrap these functionalities in a `CoreNLPClient` class. Therefore, we need to start by importing this class from Stanza.
"""
logger.info("## 2. Annotating Text with CoreNLP Interface")


"""
After the import is done, we can construct a `CoreNLPClient` instance. The constructor method takes a Python list of annotator names as argument. Here let's explore some basic annotators including tokenization, sentence split, part-of-speech tagging, lemmatization and named entity recognition (NER). 

Additionally, the client constructor accepts a `memory` argument, which specifies how much memory will be allocated to the background Java process. An `endpoint` option can be used to specify a port number used by the communication between the server and the client. The default port is 9000. However, since this port is pre-occupied by a system process in Colab, we'll manually set it to 9001 in the following example.

Also, here we manually set `be_quiet=True` to avoid an IO issue in colab notebook. You should be able to use `be_quiet=False` on your own computer, which will print detailed logging information from CoreNLP during usage.

For more options in constructing the clients, please refer to the [CoreNLP Client Options List](https://stanfordnlp.github.io/stanza/corenlp_client.html#corenlp-client-options).
"""
logger.info("After the import is done, we can construct a `CoreNLPClient` instance. The constructor method takes a Python list of annotator names as argument. Here let's explore some basic annotators including tokenization, sentence split, part-of-speech tagging, lemmatization and named entity recognition (NER).")

client = CoreNLPClient(
    annotators=['tokenize','ssplit', 'pos', 'lemma', 'ner'],
    memory='4G',
    endpoint='http://localhost:9001',
    be_quiet=True)
logger.debug(client)

client.start()

"""
After the above code block finishes executing, if you print the background processes, you should be able to find the Java CoreNLP server running.
"""
logger.info("After the above code block finishes executing, if you print the background processes, you should be able to find the Java CoreNLP server running.")

# !ps -o pid,cmd | grep java

"""
### Annotating Text

Annotating a piece of text is as simple as passing the text into an `annotate` function of the client object. After the annotation is complete, a `Document`  object will be returned with all annotations.

Note that although in general annotations are very fast, the first annotation might take a while to complete in the notebook. Please stay patient.
"""
logger.info("### Annotating Text")

text = "Albert Einstein was a German-born theoretical physicist. He developed the theory of relativity."
document = client.annotate(text)
logger.debug(type(document))

"""
## 3. Accessing Annotations

Annotations can be accessed from the returned `Document` object.

A `Document` contains a list of `Sentence`s, which contain a list of `Token`s. Here let's first explore the annotations stored in all tokens.
"""
logger.info("## 3. Accessing Annotations")

logger.debug("{:12s}\t{:12s}\t{:6s}\t{}".format("Word", "Lemma", "POS", "NER"))

for i, sent in enumerate(document.sentence):
    logger.debug("[Sentence {}]".format(i+1))
    for t in sent.token:
        logger.debug("{:12s}\t{:12s}\t{:6s}\t{}".format(t.word, t.lemma, t.pos, t.ner))
    logger.debug("")

"""
Alternatively, you can also browse the NER results by iterating over entity mentions over the sentences. For example:
"""
logger.info("Alternatively, you can also browse the NER results by iterating over entity mentions over the sentences. For example:")

logger.debug("{:30s}\t{}".format("Mention", "Type"))

for sent in document.sentence:
    for m in sent.mentions:
        logger.debug("{:30s}\t{}".format(m.entityMentionText, m.entityType))

"""
To print all annotations a sentence, token or mention has, you can simply print the corresponding obejct.
"""
logger.info("To print all annotations a sentence, token or mention has, you can simply print the corresponding obejct.")

logger.debug(document.sentence[0].token[0])

logger.debug(document.sentence[0].mentions[0])

"""
**Note**: Since the Stanza CoreNLP client interface simply ports the CoreNLP annotation results to native Python objects, for a comprehensive lists of available annotators and how their annotation results can be accessed, you will need to visit the [Stanford CoreNLP website](https://stanfordnlp.github.io/CoreNLP/).

## 4. Shutting Down the CoreNLP Server

To shut down the background CoreNLP server process, simply call the `stop` function of the client. Note that once a server is shutdown, you'll have to restart the server with the `start()` function before any annotation is requested.
"""
logger.info("## 4. Shutting Down the CoreNLP Server")

client.stop()

time.sleep(10)
# !ps -o pid,cmd | grep java

"""
### More Information

For more information on how to use the `CoreNLPClient`, please go to the [CoreNLPClient documentation page](https://stanfordnlp.github.io/stanza/corenlp_client.html).

## 5. Simplifying Client Usage with the Python `with` statement

In the above demo, we explicitly called the `client.start()` and `client.stop()` functions to start and stop a client-server connection. However, doing this in practice is usually suboptimal, since you may forget to call the `stop()` function at the end, resulting in an unused server process occupying your machine memory.

To solve is, a simple solution is to use the client interface with the [Python `with` statement](https://docs.python.org/3/reference/compound_stmts.html#the-with-statement). The `with` statement provides an elegant way to automatically start and stop the server process in your Python program, without you needing to worry about this. The following code snippet demonstrates how to establish a client, annotate an example text and then stop the server with a simple `with` statement. Note that we **always recommend** you to use the `with` statement when working with the Stanza CoreNLP client interface.
"""
logger.info("### More Information")

logger.debug("Starting a server with the Python \"with\" statement...")
with CoreNLPClient(annotators=['tokenize','ssplit', 'pos', 'lemma', 'ner'],
                   memory='4G', endpoint='http://localhost:9001', be_quiet=True) as client:
    text = "Albert Einstein was a German-born theoretical physicist."
    document = client.annotate(text)

    logger.debug("{:30s}\t{}".format("Mention", "Type"))
    for sent in document.sentence:
        for m in sent.mentions:
            logger.debug("{:30s}\t{}".format(m.entityMentionText, m.entityType))

logger.debug("\nThe server should be stopped upon exit from the \"with\" statement.")

"""
## 6. Other Resources

- [Stanza Homepage](https://stanfordnlp.github.io/stanza/)
- [FAQs](https://stanfordnlp.github.io/stanza/faq.html)
- [GitHub Repo](https://github.com/stanfordnlp/stanza)
- [Reporting Issues](https://github.com/stanfordnlp/stanza/issues)
"""
logger.info("## 6. Other Resources")

logger.info("\n\n[DONE]", bright=True)