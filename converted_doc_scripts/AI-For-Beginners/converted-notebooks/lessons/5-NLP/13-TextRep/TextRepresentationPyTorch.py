from jet.logger import logger
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader
import collections
import numpy as np
import os
import shutil
import torch
import torchtext


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
# Text classification task

As we have mentioned, we will focus on simple text classification task based on **AG_NEWS** dataset, which is to classify news headlines into one of 4 categories: World, Sports, Business and Sci/Tech.

## The Dataset

This dataset is built into [`torchtext`](https://github.com/pytorch/text) module, so we can easily access it.
"""
logger.info("# Text classification task")

os.makedirs('./data',exist_ok=True)
train_dataset, test_dataset = torchtext.datasets.AG_NEWS(root='./data')
classes = ['World', 'Sports', 'Business', 'Sci/Tech']

"""
Here, `train_dataset` and `test_dataset` contain collections that return pairs of label (number of class) and text respectively, for example:
"""
logger.info("Here, `train_dataset` and `test_dataset` contain collections that return pairs of label (number of class) and text respectively, for example:")

list(train_dataset)[0]

"""
So, let's print out the first 10 new headlines from our dataset:
"""
logger.info("So, let's print out the first 10 new headlines from our dataset:")

for i,x in zip(range(5),train_dataset):
    logger.debug(f"**{classes[x[0]]}** -> {x[1]}")

"""
Because datasets are iterators, if we want to use the data multiple times we need to convert it to list:
"""
logger.info("Because datasets are iterators, if we want to use the data multiple times we need to convert it to list:")

train_dataset, test_dataset = torchtext.datasets.AG_NEWS(root='./data')
train_dataset = list(train_dataset)
test_dataset = list(test_dataset)

"""
## Tokenization

Now we need to convert text into **numbers** that can be represented as tensors. If we want word-level representation, we need to do two things:
* use **tokenizer** to split text into **tokens**
* build a **vocabulary** of those tokens.
"""
logger.info("## Tokenization")

tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
tokenizer('He said: hello')

counter = collections.Counter()
for (label, line) in train_dataset:
    counter.update(tokenizer(line))
vocab = torchtext.vocab.vocab(counter, min_freq=1)

"""
Using vocabulary, we can easily encode out tokenized string into a set of numbers:
"""
logger.info("Using vocabulary, we can easily encode out tokenized string into a set of numbers:")

vocab_size = len(vocab)
logger.debug(f"Vocab size if {vocab_size}")

stoi = vocab.get_stoi() # dict to convert tokens to indices

def encode(x):
    return [stoi[s] for s in tokenizer(x)]

encode('I love to play with my words')

"""
## Bag of Words text representation

Because words represent meaning, sometimes we can figure out the meaning of a text by just looking at the individual words, regardless of their order in the sentence. For example, when classifying news, words like *weather*, *snow* are likely to indicate *weather forecast*, while words like *stocks*, *dollar* would count towards *financial news*.

**Bag of Words** (BoW) vector representation is the most commonly used traditional vector representation. Each word is linked to a vector index, vector element contains the number of occurrences of a word in a given document.

![Image showing how a bag of words vector representation is represented in memory.](images/bag-of-words-example.png) 

> **Note**: You can also think of BoW as a sum of all one-hot-encoded vectors for individual words in the text.

Below is an example of how to generate a bag of word representation using the Scikit Learn python library:
"""
logger.info("## Bag of Words text representation")

vectorizer = CountVectorizer()
corpus = [
        'I like hot dogs.',
        'The dog ran fast.',
        'Its hot outside.',
    ]
vectorizer.fit_transform(corpus)
vectorizer.transform(['My dog likes hot dogs on a hot day.']).toarray()

"""
To compute bag-of-words vector from the vector representation of our AG_NEWS dataset, we can use the following function:
"""
logger.info("To compute bag-of-words vector from the vector representation of our AG_NEWS dataset, we can use the following function:")

vocab_size = len(vocab)

def to_bow(text,bow_vocab_size=vocab_size):
    res = torch.zeros(bow_vocab_size,dtype=torch.float32)
    for i in encode(text):
        if i<bow_vocab_size:
            res[i] += 1
    return res

logger.debug(to_bow(train_dataset[0][1]))

"""
> **Note:** Here we are using global `vocab_size` variable to specify default size of the vocabulary. Since often vocabulary size is pretty big, we can limit the size of the vocabulary to most frequent words. Try lowering `vocab_size` value and running the code below, and see how it affects the accuracy. You should expect some accuracy drop, but not dramatic, in lieu of higher performance.

## Training BoW classifier

Now that we have learned how to build Bag-of-Words representation of our text, let's train a classifier on top of it. First, we need to convert our dataset for training in such a way, that all positional vector representations are converted to bag-of-words representation. This can be achieved by passing `bowify` function as `collate_fn` parameter to standard torch `DataLoader`:
"""
logger.info("## Training BoW classifier")


def bowify(b):
    return (
            torch.LongTensor([t[0]-1 for t in b]),
            torch.stack([to_bow(t[1]) for t in b])
    )

train_loader = DataLoader(train_dataset, batch_size=16, collate_fn=bowify, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, collate_fn=bowify, shuffle=True)

"""
Now let's define a simple classifier neural network that contains one linear layer. The size of the input vector equals to `vocab_size`, and output size corresponds to the number of classes (4). Because we are solving classification task, the final activation function is `LogSoftmax()`.
"""
logger.info("Now let's define a simple classifier neural network that contains one linear layer. The size of the input vector equals to `vocab_size`, and output size corresponds to the number of classes (4). Because we are solving classification task, the final activation function is `LogSoftmax()`.")

net = torch.nn.Sequential(torch.nn.Linear(vocab_size,4),torch.nn.LogSoftmax(dim=1))

"""
Now we will define standard PyTorch training loop. Because our dataset is quite large, for our teaching purpose we will train only for one epoch, and sometimes even for less than an epoch (specifying the `epoch_size` parameter allows us to limit training). We would also report accumulated training accuracy during training; the frequency of reporting is specified using `report_freq` parameter.
"""
logger.info("Now we will define standard PyTorch training loop. Because our dataset is quite large, for our teaching purpose we will train only for one epoch, and sometimes even for less than an epoch (specifying the `epoch_size` parameter allows us to limit training). We would also report accumulated training accuracy during training; the frequency of reporting is specified using `report_freq` parameter.")

def train_epoch(net,dataloader,lr=0.01,optimizer=None,loss_fn = torch.nn.NLLLoss(),epoch_size=None, report_freq=200):
    optimizer = optimizer or torch.optim.Adam(net.parameters(),lr=lr)
    net.train()
    total_loss,acc,count,i = 0,0,0,0
    for labels,features in dataloader:
        optimizer.zero_grad()
        out = net(features)
        loss = loss_fn(out,labels) #cross_entropy(out,labels)
        loss.backward()
        optimizer.step()
        total_loss+=loss
        _,predicted = torch.max(out,1)
        acc+=(predicted==labels).sum()
        count+=len(labels)
        i+=1
        if i%report_freq==0:
            logger.debug(f"{count}: acc={acc.item()/count}")
        if epoch_size and count>epoch_size:
            break
    return total_loss.item()/count, acc.item()/count

train_epoch(net,train_loader,epoch_size=15000)

"""
## BiGrams, TriGrams and N-Grams

One limitation of a bag of words approach is that some words are part of multi word expressions, for example, the word 'hot dog' has a completely different meaning than the words 'hot' and 'dog' in other contexts. If we represent words 'hot` and 'dog' always by the same vectors, it can confuse our model.

To address this, **N-gram representations** are often used in methods of document classification, where the frequency of each word, bi-word or tri-word is a useful feature for training classifiers. In bigram representation, for example, we will add all word pairs to the vocabulary, in addition to original words. 

Below is an example of how to generate a bigram bag of word representation using the Scikit Learn:
"""
logger.info("## BiGrams, TriGrams and N-Grams")

bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)
corpus = [
        'I like hot dogs.',
        'The dog ran fast.',
        'Its hot outside.',
    ]
bigram_vectorizer.fit_transform(corpus)
logger.debug("Vocabulary:\n",bigram_vectorizer.vocabulary_)
bigram_vectorizer.transform(['My dog likes hot dogs on a hot day.']).toarray()

"""
The main drawback of N-gram approach is that vocabulary size starts to grow extremely fast. In practice, we need to combine N-gram representation with some dimensionality reduction techniques, such as *embeddings*, which we will discuss in the next unit.

To use N-gram representation in our **AG News** dataset, we need to build special ngram vocabulary:
"""
logger.info("The main drawback of N-gram approach is that vocabulary size starts to grow extremely fast. In practice, we need to combine N-gram representation with some dimensionality reduction techniques, such as *embeddings*, which we will discuss in the next unit.")

counter = collections.Counter()
for (label, line) in train_dataset:
    l = tokenizer(line)
    counter.update(torchtext.data.utils.ngrams_iterator(l,ngrams=2))

bi_vocab = torchtext.vocab.vocab(counter, min_freq=1)

logger.debug("Bigram vocabulary length = ",len(bi_vocab))

"""
We could then use the same code as above to train the classifier, however, it would be very memory-inefficient. In the next unit, we will train bigram classifier using embeddings.

> **Note:** You can only leave those ngrams that occur in the text more than specified number of times. This will make sure that infrequent bigrams will be omitted, and will decrease the dimensionality significantly. To do this, set `min_freq` parameter to a higher value, and observe the length of vocabulary change.

## Term Frequency Inverse Document Frequency TF-IDF

In BoW representation, word occurrences are evenly weighted, regardless of the word itself. However, it is clear that frequent words, such as *a*, *in*, etc. are much less important for the classification, than specialized terms. In fact, in most NLP tasks some words are more relevant than others.

**TF-IDF** stands for **term frequency–inverse document frequency**. It is a variation of bag of words, where instead of a binary 0/1 value indicating the appearance of a word in a document, a floating-point value is used, which is related to the frequency of word occurrence in the corpus.

More formally, the weight $w_{ij}$ of a word $i$ in the document $j$ is defined as:
$$
w_{ij} = tf_{ij}\times\log({N\over df_i})
$$
where
* $tf_{ij}$ is the number of occurrences of $i$ in $j$, i.e. the BoW value we have seen before
* $N$ is the number of documents in the collection
* $df_i$ is the number of documents containing the word $i$ in the whole collection

TF-IDF value $w_{ij}$ increases proportionally to the number of times a word appears in a document and is offset by the number of documents in the corpus that contains the word, which helps to adjust for the fact that some words appear more frequently than others. For example, if the word appears in *every* document in the collection, $df_i=N$, and $w_{ij}=0$, and those terms would be completely disregarded.

You can easily create TF-IDF vectorization of text using Scikit Learn:
"""
logger.info("## Term Frequency Inverse Document Frequency TF-IDF")

vectorizer = TfidfVectorizer(ngram_range=(1,2))
vectorizer.fit_transform(corpus)
vectorizer.transform(['My dog likes hot dogs on a hot day.']).toarray()

"""
## Conclusion 

However even though TF-IDF representations provide frequency weight to different words they are unable to represent meaning or order. As the famous linguist J. R. Firth said in 1935, “The complete meaning of a word is always contextual, and no study of meaning apart from context can be taken seriously.”. We will learn later in the course how to capture contextual information from text using language modeling.
"""
logger.info("## Conclusion")

logger.info("\n\n[DONE]", bright=True)