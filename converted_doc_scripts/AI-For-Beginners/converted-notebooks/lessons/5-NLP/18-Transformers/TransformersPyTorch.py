from jet.logger import logger
from torchnlp import *
import os
import shutil
import torch
import torchtext
import transformers


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
# Attention mechanisms and transformers

One major drawback of recurrent networks is that all words in a sequence have the same impact on the result. This causes sub-optimal performance with standard LSTM encoder-decoder models for sequence to sequence tasks, such as Named Entity Recognition and Machine Translation. In reality specific words in the input sequence often have more impact on sequential outputs than others.

Consider sequence-to-sequence model, such as machine translation. It is implemented by two recurrent networks, where one network (**encoder**) would collapse input sequence into hidden state, and another one, **decoder**, would unroll this hidden state into translated result. The problem with this approach is that final state of the network would have hard time remembering the beginning of a sentence, thus causing poor quality of the model on long sentences.

**Attention Mechanisms** provide a means of weighting the contextual impact of each input vector on each output prediction of the RNN. The way it is implemented is by creating shortcuts between intermediate states of the input RNN, and output RNN. In this manner, when generating output symbol $y_t$, we will take into account all input hidden states $h_i$, with different weight coefficients $\alpha_{t,i}$. 

![Image showing an encoder/decoder model with an additive attention layer](./images/encoder-decoder-attention.png)
*The encoder-decoder model with additive attention mechanism in [Bahdanau et al., 2015](https://arxiv.org/pdf/1409.0473.pdf), cited from [this blog post](https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html)*

Attention matrix $\{\alpha_{i,j}\}$ would represent the degree which certain input words play in generation of a given word in the output sequence. Below is the example of such a matrix:

![Image showing a sample alignment found by RNNsearch-50, taken from Bahdanau - arviz.org](./images/bahdanau-fig3.png)

*Figure taken from [Bahdanau et al., 2015](https://arxiv.org/pdf/1409.0473.pdf) (Fig.3)*

Attention mechanisms are responsible for much of the current or near current state of the art in Natural language processing. Adding attention however greatly increases the number of model parameters which led to scaling issues with RNNs. A key constraint of scaling RNNs is that the recurrent nature of the models makes it challenging to batch and parallelize training. In an RNN each element of a sequence needs to be processed in sequential order which means it cannot be easily parallelized.

Adoption of attention mechanisms combined with this constraint led to the creation of the now State of the Art Transformer Models that we know and use today from BERT to OpenGPT3.

## Transformer models

Instead of forwarding the context of each previous prediction into the next evaluation step, **transformer models** use **positional encodings** and attention to capture the context of a given input with in a provided window of text. The image below shows how positional encodings with attention can capture context within a given window.

![Animated GIF showing how the evaluations are performed in transformer models.](./images/transformer-animated-explanation.gif) 

Since each input position is mapped independently to each output position, transformers can parallelize better than RNNs, which enables much larger and more expressive language models. Each attention head can be used to learn different relationships between words that improves downstream Natural Language Processing tasks.

**BERT** (Bidirectional Encoder Representations from Transformers) is a very large multi layer transformer network with 12 layers for *BERT-base*, and 24 for *BERT-large*. The model is first pre-trained on large corpus of text data (WikiPedia + books) using unsupervised training (predicting masked words in a sentence). During pre-training the model absorbs significant level of language understanding which can then be leveraged with other datasets using fine tuning. This process is called **transfer learning**. 

![picture from http://jalammar.github.io/illustrated-bert/](./images/jalammarBERT-language-modeling-masked-lm.png)

There are many variations of Transformer architectures including BERT, DistilBERT. BigBird, OpenGPT3 and more that can be fine tuned. The [HuggingFace package](https://github.com/huggingface/) provides repository for training many of these architectures with PyTorch. 

## Using BERT for text classification

Let's see how we can use pre-trained BERT model for solving our traditional task: sequence classification. We will classify our original AG News dataset.

First, let's load HuggingFace library and our dataset:
"""
logger.info("# Attention mechanisms and transformers")

train_dataset, test_dataset, classes, vocab = load_dataset()
vocab_len = len(vocab)

"""
Because we will be using pre-trained BERT model, we would need to use specific tokenizer. First, we will load a tokenizer associated with pre-trained BERT model.

HuggingFace library contains a repository of pre-trained models, which you can use just by specifying their names as arguments to `from_pretrained` functions. All required binary files for the model would automatically be downloaded.

However, at certain times you would need to load your own models, in which case you can specify the directory that contains all relevant files, including parameters for tokenizer, `config.json` file with model parameters, binary weights, etc.
"""
logger.info("Because we will be using pre-trained BERT model, we would need to use specific tokenizer. First, we will load a tokenizer associated with pre-trained BERT model.")

bert_model = 'bert-base-uncased'

bert_model = './bert'

tokenizer = transformers.BertTokenizer.from_pretrained(bert_model)

MAX_SEQ_LEN = 128
PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

"""
The `tokenizer` object contains the `encode` function that can be directly used to encode text:
"""
logger.info("The `tokenizer` object contains the `encode` function that can be directly used to encode text:")

tokenizer.encode('PyTorch is a great framework for NLP')

"""
Then, let's create iterators which we will use during training to access the data. Because BERT uses it's own encoding function, we would need to define a padding function similar to `padify` we have defined before:
"""
logger.info("Then, let's create iterators which we will use during training to access the data. Because BERT uses it's own encoding function, we would need to define a padding function similar to `padify` we have defined before:")

def pad_bert(b):
    v = [tokenizer.encode(x[1]) for x in b]
    l = max(map(len,v))
    return ( # tuple of two tensors - labels and features
        torch.LongTensor([t[0] for t in b]),
        torch.stack([torch.nn.functional.pad(torch.tensor(t),(0,l-len(t)),mode='constant',value=0) for t in v])
    )

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, collate_fn=pad_bert, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, collate_fn=pad_bert)

"""
In our case, we will be using pre-trained BERT model called `bert-base-uncased`. Let's load the model using `BertForSequenceClassfication` package. This ensures that our model already has a required architecture for classification, including final classifier. You will see warning message stating that weights of the final classifier are not initialized, and model would require pre-training - that is perfectly okay, because it is exactly what we are about to do!
"""
logger.info("In our case, we will be using pre-trained BERT model called `bert-base-uncased`. Let's load the model using `BertForSequenceClassfication` package. This ensures that our model already has a required architecture for classification, including final classifier. You will see warning message stating that weights of the final classifier are not initialized, and model would require pre-training - that is perfectly okay, because it is exactly what we are about to do!")

model = transformers.BertForSequenceClassification.from_pretrained(bert_model,num_labels=4).to(device)

"""
Now we are ready to begin training! Because BERT is already pre-trained, we want to start with rather small learning rate in order not to destroy initial weights.

All hard work is done by `BertForSequenceClassification` model. When we call the model on the training data, it returns both loss and network output for input minibatch. We use loss for parameter optimization (`loss.backward()` does the backward pass), and `out` for computing training accuracy by comparing obtained labels `labs` (computed using `argmax`) with expected `labels`.

In order to control the process, we accumulate loss and accuracy over several iterations, and print them every `report_freq` training cycles.

This training will likely take quite a long time, so we limit the number of iterations.
"""
logger.info("Now we are ready to begin training! Because BERT is already pre-trained, we want to start with rather small learning rate in order not to destroy initial weights.")

optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

report_freq = 50
iterations = 500 # make this larger to train for longer time!

model.train()

i,c = 0,0
acc_loss = 0
acc_acc = 0

for labels,texts in train_loader:
    labels = labels.to(device)-1 # get labels in the range 0-3
    texts = texts.to(device)
    loss, out = model(texts, labels=labels)[:2]
    labs = out.argmax(dim=1)
    acc = torch.mean((labs==labels).type(torch.float32))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    acc_loss += loss
    acc_acc += acc
    i+=1
    c+=1
    if i%report_freq==0:
        logger.debug(f"Loss = {acc_loss.item()/c}, Accuracy = {acc_acc.item()/c}")
        c = 0
        acc_loss = 0
        acc_acc = 0
    iterations-=1
    if not iterations:
        break

"""
You can see (especially if you increase the number of iterations and wait long enough) that BERT classification gives us pretty good accuracy! That is because BERT already understands quite well the structure of the language, and we only need to fine-tune final classifier. However, because BERT is a large model, the whole training process takes a long time, and requires serious computational power! (GPU, and preferably more than one).

> **Note:** In our example, we have been using one of the smallest pre-trained BERT models. There are larger models that are likely to yield better results.

## Evaluating the model performance

Now we can evaluate performance of our model on test dataset. Evaluation loop is pretty similar to training loop, but we should not forget to switch model to evaluation mode by calling `model.eval()`.
"""
logger.info("## Evaluating the model performance")

model.eval()
iterations = 100
acc = 0
i = 0
for labels,texts in test_loader:
    labels = labels.to(device)-1
    texts = texts.to(device)
    _, out = model(texts, labels=labels)[:2]
    labs = out.argmax(dim=1)
    acc += torch.mean((labs==labels).type(torch.float32))
    i+=1
    if i>iterations: break

logger.debug(f"Final accuracy: {acc.item()/i}")

"""
## Takeaway

In this unit, we have seen how easy it is to take pre-trained language model from **transformers** library and adapt it to our text classification task. Similarly, BERT models can be used for entity extraction, question answering, and other NLP tasks.

Transformer models represent current state-of-the-art in NLP, and in most of the cases it should be the first solution you start experimenting with when implementing custom NLP solutions. However, understanding basic underlying principles of recurrent neural networks discussed in this module is extremely important if you want to build advanced neural models.
"""
logger.info("## Takeaway")

logger.info("\n\n[DONE]", bright=True)