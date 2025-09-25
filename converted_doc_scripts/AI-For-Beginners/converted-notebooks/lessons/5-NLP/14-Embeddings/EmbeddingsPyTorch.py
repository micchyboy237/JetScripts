from jet.logger import logger
from torchnlp import *
import gensim.downloader as api
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
## Embeddings

In our previous example, we operated on high-dimensional bag-of-words vectors with length `vocab_size`, and we were explicitly converting from low-dimensional positional representation vectors into sparse one-hot representation. This one-hot representation is not memory-efficient, in addition, each word is treated independently from each other, i.e. one-hot encoded vectors do not express any semantic similarity between words.

In this unit, we will continue exploring **News AG** dataset. To begin, let's load the data and get some definitions from the previous notebook.
"""
logger.info("## Embeddings")

train_dataset, test_dataset, classes, vocab = load_dataset()
vocab_size = len(vocab)
logger.debug("Vocab size = ",vocab_size)

"""
## What is embedding?

The idea of **embedding** is to represent words by lower-dimensional dense vectors, which somehow reflect semantic meaning of a word. We will later discuss how to build meaningful word embeddings, but for now let's just think of embeddings as a way to lower dimensionality of a word vector. 

So, embedding layer would take a word as an input, and produce an output vector of specified `embedding_size`. In a sense, it is very similar to `Linear` layer, but instead of taking one-hot encoded vector, it will be able to take a word number as an input.

By using embedding layer as a first layer in our network, we can switch from bag-of-words to **embedding bag** model, where we first convert each word in our text into corresponding embedding, and then compute some aggregate function over all those embeddings, such as `sum`, `average` or `max`.  

![Image showing an embedding classifier for five sequence words.](images/embedding-classifier-example.png)

Our classifier neural network will start with embedding layer, then aggregation layer, and linear classifier on top of it:
"""
logger.info("## What is embedding?")

class EmbedClassifier(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.fc = torch.nn.Linear(embed_dim, num_class)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x,dim=1)
        return self.fc(x)

"""
### Dealing with variable sequence size

As a result of this architecture, minibatches to our network would need to be created in a certain way. In the previous unit, when using bag-of-words, all BoW tensors in a minibatch had equal size `vocab_size`, regardless of the actual length of our text sequence. Once we move to word embeddings, we would end up with variable number of words in each text sample, and when combining those samples into minibatches we would have to apply some padding.

This can be done using the same technique of providing `collate_fn` function to the datasource:
"""
logger.info("### Dealing with variable sequence size")

def padify(b):
    v = [encode(x[1]) for x in b]
    l = max(map(len,v))
    return ( # tuple of two tensors - labels and features
        torch.LongTensor([t[0]-1 for t in b]),
        torch.stack([torch.nn.functional.pad(torch.tensor(t),(0,l-len(t)),mode='constant',value=0) for t in v])
    )

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, collate_fn=padify, shuffle=True)

"""
### Training embedding classifier

Now that we have defined proper dataloader, we can train the model using the training function we have defined in the previous unit:
"""
logger.info("### Training embedding classifier")

net = EmbedClassifier(vocab_size,32,len(classes)).to(device)
train_epoch(net,train_loader, lr=1, epoch_size=25000)

"""
> **Note**: We are only training for 25k records here (less than one full epoch) for the sake of time, but you can continue training, write a function to train for several epochs, and experiment with learning rate parameter to achieve higher accuracy. You should be able to go to the accuracy of about 90%.

### EmbeddingBag Layer and Variable-Length Sequence Representation

In the previous architecture, we needed to pad all sequences to the same length in order to fit them into a minibatch. This is not the most efficient way to represent variable length sequences - another apporach would be to use **offset** vector, which would hold offsets of all sequences stored in one large vector.

![Image showing an offset sequence representation](images/offset-sequence-representation.png)

> **Note**: On the picture above, we show a sequence of characters, but in our example we are working with sequences of words. However, the general principle of representing sequences with offset vector remains the same.

To work with offset representation, we use [`EmbeddingBag`](https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html) layer. It is similar to `Embedding`, but it takes content vector and offset vector as input, and it also includes averaging layer, which can be `mean`, `sum` or `max`.

Here is modified network that uses `EmbeddingBag`:
"""
logger.info("### EmbeddingBag Layer and Variable-Length Sequence Representation")

class EmbedClassifier(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = torch.nn.EmbeddingBag(vocab_size, embed_dim)
        self.fc = torch.nn.Linear(embed_dim, num_class)

    def forward(self, text, off):
        x = self.embedding(text, off)
        return self.fc(x)

"""
To prepare the dataset for training, we need to provide a conversion function that will prepare the offset vector:
"""
logger.info("To prepare the dataset for training, we need to provide a conversion function that will prepare the offset vector:")

def offsetify(b):
    x = [torch.tensor(encode(t[1])) for t in b]
    o = [0] + [len(t) for t in x]
    o = torch.tensor(o[:-1]).cumsum(dim=0)
    return (
        torch.LongTensor([t[0]-1 for t in b]), # labels
        torch.cat(x), # text
        o
    )

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, collate_fn=offsetify, shuffle=True)

"""
Note, that unlike in all previous examples, our network now accepts two parameters: data vector and offset vector, which are of different sizes. Similarly, our data loader also provides us with 3 values instead of 2: both text and offset vectors are provided as features. Therefore, we need to slightly adjust our training function to take care of that:
"""
logger.info("Note, that unlike in all previous examples, our network now accepts two parameters: data vector and offset vector, which are of different sizes. Similarly, our data loader also provides us with 3 values instead of 2: both text and offset vectors are provided as features. Therefore, we need to slightly adjust our training function to take care of that:")

net = EmbedClassifier(vocab_size,32,len(classes)).to(device)

def train_epoch_emb(net,dataloader,lr=0.01,optimizer=None,loss_fn = torch.nn.CrossEntropyLoss(),epoch_size=None, report_freq=200):
    optimizer = optimizer or torch.optim.Adam(net.parameters(),lr=lr)
    loss_fn = loss_fn.to(device)
    net.train()
    total_loss,acc,count,i = 0,0,0,0
    for labels,text,off in dataloader:
        optimizer.zero_grad()
        labels,text,off = labels.to(device), text.to(device), off.to(device)
        out = net(text, off)
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


train_epoch_emb(net,train_loader, lr=4, epoch_size=25000)

"""
## Semantic Embeddings: Word2Vec

In our previous example, the model embedding layer learnt to map words to vector representation, however, this representation did not have much semantical meaning. It would be nice to learn such vector representation, that similar words or synonyms would correspond to vectors that are close to each other in terms of some vector distance (eg. euclidian distance).

To do that, we need to pre-train our embedding model on a large collection of text in a specific way. One of the first ways to train semantic embeddings is called [Word2Vec](https://en.wikipedia.org/wiki/Word2vec). It is based on two main architectures that are used to produce a distributed representation of words:

 - **Continuous bag-of-words** (CBoW) — in this architecture, we train the model to predict a word from surrounding context. Given the ngram $(W_{-2},W_{-1},W_0,W_1,W_2)$, the goal of the model is to predict $W_0$ from $(W_{-2},W_{-1},W_1,W_2)$.
 - **Continuous skip-gram** is opposite to CBoW. The model uses surrounding window of context words to predict the current word.

CBoW is faster, while skip-gram is slower, but does a better job of representing infrequent words.

![Image showing both CBoW and Skip-Gram algorithms to convert words to vectors.](./images/example-algorithms-for-converting-words-to-vectors.png)

To experiment with word2vec embedding pre-trained on Google News dataset, we can use **gensim** library. Below we find the words most similar to 'neural'

> **Note:** When you first create word vectors, downloading them can take some time!
"""
logger.info("## Semantic Embeddings: Word2Vec")

w2v = api.load('word2vec-google-news-300')

for w,p in w2v.most_similar('neural'):
    logger.debug(f"{w} -> {p}")

"""
We can also compute vector embeddings from the word, to be used in training classification model (we only show first 20 components of the vector for clarity):
"""
logger.info("We can also compute vector embeddings from the word, to be used in training classification model (we only show first 20 components of the vector for clarity):")

w2v.word_vec('play')[:20]

"""
Great thing about semantical embeddings is that you can manipulate vector encoding to change the semantics. For example, we can ask to find a word, whose vector representation would be as close as possible to words *king* and *woman*, and as far away from the word *man*:
"""
logger.info("Great thing about semantical embeddings is that you can manipulate vector encoding to change the semantics. For example, we can ask to find a word, whose vector representation would be as close as possible to words *king* and *woman*, and as far away from the word *man*:")

w2v.most_similar(positive=['king','woman'],negative=['man'])[0]

"""
Both CBoW and Skip-Grams are “predictive” embeddings, in that they only take local contexts into account. Word2Vec does not take advantage of global context. 

**FastText**, builds on Word2Vec by learning vector representations for each word and the charachter n-grams found within each word. The values of the representations are then averaged into one vector at each training step. While this adds a lot of additional computation to pre-training it enables word embeddings to encode sub-word information. 

Another method, **GloVe**, leverages the idea of co-occurence matrix, uses neural methods to decompose co-occurrence matrix into more expressive and non linear word vectors.

You can play with the example by changing embeddings to FastText and GloVe, since gensim supports several different word embedding models.

## Using Pre-Trained Embeddings in PyTorch

We can modify the example above to pre-populate the matrix in our embedding layer with semantical embeddings, such as Word2Vec. We need to take into account that vocabularies of pre-trained embedding and our text corpus will likely not match, so we will initialize weights for the missing words with random values:
"""
logger.info("## Using Pre-Trained Embeddings in PyTorch")

embed_size = len(w2v.get_vector('hello'))
logger.debug(f'Embedding size: {embed_size}')

net = EmbedClassifier(vocab_size,embed_size,len(classes))

logger.debug('Populating matrix, this will take some time...',end='')
found, not_found = 0,0
for i,w in enumerate(vocab.get_itos()):
    try:
        net.embedding.weight[i].data = torch.tensor(w2v.get_vector(w))
        found+=1
    except:
        net.embedding.weight[i].data = torch.normal(0.0,1.0,(embed_size,))
        not_found+=1

logger.debug(f"Done, found {found} words, {not_found} words missing")
net = net.to(device)

"""
Now let's train our model. Note that the time it takes to train the model is significantly larger than in the previous example, due to larger embedding layer size, and thus much higher number of parameters. Also, because of this, we may need to train our model on more examples if we want to avoid overfitting.
"""
logger.info("Now let's train our model. Note that the time it takes to train the model is significantly larger than in the previous example, due to larger embedding layer size, and thus much higher number of parameters. Also, because of this, we may need to train our model on more examples if we want to avoid overfitting.")

train_epoch_emb(net,train_loader, lr=4, epoch_size=25000)

"""
In our case we do not see huge increase in accuracy, which is likely to quite different vocabularies. 
To overcome the problem of different vocabularies, we can use one of the following solutions:
* Re-train word2vec model on our vocabulary
* Load our dataset with the vocabulary from the pre-trained word2vec model. Vocabulary used to load the dataset can be specified during loading.

The latter approach seems easiter, especially because PyTorch `torchtext` framework contains built-in support for embeddings. We can, for example, instantiate GloVe-based vocabulary in the following manner:
"""
logger.info("In our case we do not see huge increase in accuracy, which is likely to quite different vocabularies.")

vocab = torchtext.vocab.GloVe(name='6B', dim=50)

"""
Loaded vocabulary has the following basic operations:
* `vocab.stoi` dictionary allows us to convert word into its dictionary index
* `vocab.itos` does the opposite - converts number into word
* `vocab.vectors` is the array of embedding vectors, so to get the embedding of a word `s` we need to use `vocab.vectors[vocab.stoi[s]]`

Here is the example of manipulating embeddings to demonstrate the equation **kind-man+woman = queen** (I had to tweak the coefficient a bit to make it work):
"""
logger.info("Loaded vocabulary has the following basic operations:")

qvec = vocab.vectors[vocab.stoi['king']]-vocab.vectors[vocab.stoi['man']]+1.3*vocab.vectors[vocab.stoi['woman']]
d = torch.sum((vocab.vectors-qvec)**2,dim=1)
min_idx = torch.argmin(d)
vocab.itos[min_idx]

"""
To train the classifier using those embeddings, we first need to encode our dataset using GloVe vocabulary:
"""
logger.info("To train the classifier using those embeddings, we first need to encode our dataset using GloVe vocabulary:")

def offsetify(b):
    x = [torch.tensor(encode(t[1],voc=vocab)) for t in b] # pass the instance of vocab to encode function!
    o = [0] + [len(t) for t in x]
    o = torch.tensor(o[:-1]).cumsum(dim=0)
    return (
        torch.LongTensor([t[0]-1 for t in b]), # labels
        torch.cat(x), # text
        o
    )

"""
As we have seen above, all vector embeddings are stored in `vocab.vectors` matrix. It makes it super-easy to load those weights into weights of embedding layer using simple copying:
"""
logger.info("As we have seen above, all vector embeddings are stored in `vocab.vectors` matrix. It makes it super-easy to load those weights into weights of embedding layer using simple copying:")

net = EmbedClassifier(len(vocab),len(vocab.vectors[0]),len(classes))
net.embedding.weight.data = vocab.vectors
net = net.to(device)

"""
Now let's train our model and see if we get better results:
"""
logger.info("Now let's train our model and see if we get better results:")

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, collate_fn=offsetify, shuffle=True)
train_epoch_emb(net,train_loader, lr=4, epoch_size=25000)

"""
One of the reasons we are not seeing significant increase in accuracy is due to the fact that some words from our dataset are missing in the pre-trained GloVe vocabulary, and thus they are essentially ignored. To overcome this fact, we can train our own embeddings on our dataset.

## Contextual Embeddings

One key limitation of traditional pretrained embedding representations such as Word2Vec is the problem of word sense disambiguation. While pretrained embeddings can capture some of the meaning of words in context, every possible meaning of a word is encoded into the same embedding. This can cause problems in downstream models, since many words such as the word 'play' have different meanings depending on the context they are used in.

For example word 'play' in those two different sentences have quite different meaning:
- I went to a **play** at the theature.
- John wants to **play** with his friends.

The pretrained embeddings above represent both of these meanings of the word 'play' in the same embedding. To overcome this limitation, we need to build embeddings based on the **language model**, which is trained on a large corpus of text, and *knows* how words can be put together in different contexts. Discussing contextual embeddings is out of scope for this tutorial, but we will come back to them when talking about language models in the next unit.
"""
logger.info("## Contextual Embeddings")

logger.info("\n\n[DONE]", bright=True)