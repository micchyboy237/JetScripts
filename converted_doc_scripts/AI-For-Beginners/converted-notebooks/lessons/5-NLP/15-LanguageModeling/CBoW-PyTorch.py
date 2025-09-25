from jet.logger import logger
import builtins
import collections
import numpy as np
import os
import random
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
## Training CBoW Model

This notebooks is a part of [AI for Beginners Curriculum](http://aka.ms/ai-beginners)

In this example, we will look at training CBoW language model to get our own Word2Vec embedding space. We will use AG News dataset as the source of text.
"""
logger.info("## Training CBoW Model")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
First let's load our dataset and define tokenizer and vocabulary. We will set `vocab_size` to 5000 to limit computations a bit.
"""
logger.info("First let's load our dataset and define tokenizer and vocabulary. We will set `vocab_size` to 5000 to limit computations a bit.")

def load_dataset(ngrams = 1, min_freq = 1, vocab_size = 5000 , lines_cnt = 500):
    tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
    logger.debug("Loading dataset...")
    test_dataset, train_dataset  = torchtext.datasets.AG_NEWS(root='./data')
    train_dataset = list(train_dataset)
    test_dataset = list(test_dataset)
    classes = ['World', 'Sports', 'Business', 'Sci/Tech']
    logger.debug('Building vocab...')
    counter = collections.Counter()
    for i, (_, line) in enumerate(train_dataset):
        counter.update(torchtext.data.utils.ngrams_iterator(tokenizer(line),ngrams=ngrams))
        if i == lines_cnt:
            break
    vocab = torchtext.vocab.Vocab(collections.Counter(dict(counter.most_common(vocab_size))), min_freq=min_freq)
    return train_dataset, test_dataset, classes, vocab, tokenizer

train_dataset, test_dataset, _, vocab, tokenizer = load_dataset()

def encode(x, vocabulary, tokenizer = tokenizer):
    return [vocabulary[s] for s in tokenizer(x)]

"""
## CBoW Model

CBoW learns to predict a word based on the $2N$ neighboring words. For example, when $N=1$, we will get the following pairs from the sentence *I like to train networks*: (like,I), (I, like), (to, like), (like,to), (train,to), (to, train), (networks, train), (train,networks). Here, first word is the neighboring word used as an input, and second word is the one we are predicting.

To build a network to predict next word, we will need to supply neighboring word as input, and get word number as output. The architecture of CBoW network is the following:

* Input word is passed through the embedding layer. This very embedding layer would be our Word2Vec embedding, thus we will define it separately as `embedder` variable. We will use embedding size = 30 in this example, even though you might want to experiment with higher dimensions (real word2vec has 300)
* Embedding vector would then be passed to a linear layer that will predict output word. Thus it has the `vocab_size` neurons.

For the output, if we use `CrossEntropyLoss` as loss function, we would also have to provide just word numbers as expected results, without one-hot encoding.
"""
logger.info("## CBoW Model")

vocab_size = len(vocab)

embedder = torch.nn.Embedding(num_embeddings = vocab_size, embedding_dim = 30)
model = torch.nn.Sequential(
    embedder,
    torch.nn.Linear(in_features = 30, out_features = vocab_size),
)

logger.debug(model)

"""
## Preparing Training Data

Now let's program the main function that will compute CBoW word pairs from text. This function will allow us to specify window size, and will return a set of pairs - input and output word. Note that this function can be used on words, as well as on vectors/tensors - which will allow us to encode the text, before passing it to `to_cbow` function.
"""
logger.info("## Preparing Training Data")

def to_cbow(sent,window_size=2):
    res = []
    for i,x in enumerate(sent):
        for j in range(max(0,i-window_size),min(i+window_size+1,len(sent))):
            if i!=j:
                res.append([sent[j],x])
    return res

logger.debug(to_cbow(['I','like','to','train','networks']))
logger.debug(to_cbow(encode('I like to train networks', vocab)))

"""
Let's prepare the training dataset. We will go through all news, call `to_cbow` to get the list of word pairs, and add those pairs to `X` and `Y`. For the sake of time, we will only consider first 10k news items - you can easily remove the limitation in case you have more time to wait, and want to get better embeddings :)
"""
logger.info("Let's prepare the training dataset. We will go through all news, call `to_cbow` to get the list of word pairs, and add those pairs to `X` and `Y`. For the sake of time, we will only consider first 10k news items - you can easily remove the limitation in case you have more time to wait, and want to get better embeddings :)")

X = []
Y = []
for i, x in zip(range(10000), train_dataset):
    for w1, w2 in to_cbow(encode(x[1], vocab), window_size = 5):
        X.append(w1)
        Y.append(w2)

X = torch.tensor(X)
Y = torch.tensor(Y)

"""
We will also convert that data to one dataset, and create dataloader:
"""
logger.info("We will also convert that data to one dataset, and create dataloader:")

class SimpleIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, X, Y):
        super(SimpleIterableDataset).__init__()
        self.data = []
        for i in range(len(X)):
            self.data.append( (Y[i], X[i]) )
        random.shuffle(self.data)

    def __iter__(self):
        return iter(self.data)

"""
We will also convert that data to one dataset, and create dataloader:
"""
logger.info("We will also convert that data to one dataset, and create dataloader:")

ds = SimpleIterableDataset(X, Y)
dl = torch.utils.data.DataLoader(ds, batch_size = 256)

"""
Now let's do the actual training. We will use `SGD` optimizer with pretty high learning rate. You can also try playing around with other optimizers, such as `Adam`. We will train for 10 epochs to begin with - and you can re-run this cell if you want even lower loss.
"""
logger.info("Now let's do the actual training. We will use `SGD` optimizer with pretty high learning rate. You can also try playing around with other optimizers, such as `Adam`. We will train for 10 epochs to begin with - and you can re-run this cell if you want even lower loss.")

def train_epoch(net, dataloader, lr = 0.01, optimizer = None, loss_fn = torch.nn.CrossEntropyLoss(), epochs = None, report_freq = 1):
    optimizer = optimizer or torch.optim.Adam(net.parameters(), lr = lr)
    loss_fn = loss_fn.to(device)
    net.train()

    for i in range(epochs):
        total_loss, j = 0, 0,
        for labels, features in dataloader:
            optimizer.zero_grad()
            features, labels = features.to(device), labels.to(device)
            out = net(features)
            loss = loss_fn(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss
            j += 1
        if i % report_freq == 0:
            logger.debug(f"Epoch: {i+1}: loss={total_loss.item()/j}")

    return total_loss.item()/j

train_epoch(net = model, dataloader = dl, optimizer = torch.optim.SGD(model.parameters(), lr = 0.1), loss_fn = torch.nn.CrossEntropyLoss(), epochs = 10)

"""
## Trying out Word2Vec

To use Word2Vec, let's extract vectors corresponding to all words in our vocabulary:
"""
logger.info("## Trying out Word2Vec")

vectors = torch.stack([embedder(torch.tensor(vocab[s])) for s in vocab.itos], 0)

"""
Let's see, for example, how the word **Paris** is encoded into a vector:
"""
logger.info("Let's see, for example, how the word **Paris** is encoded into a vector:")

paris_vec = embedder(torch.tensor(vocab['paris']))
logger.debug(paris_vec)

"""
It is interesting to use Word2Vec to look for synonyms. The following function will return `n` closest words to a given input. To find them, we compute the norm of $|w_i - v|$, where $v$ is the vector corresponding to our input word, and $w_i$ is the encoding of $i$-th word in the vocabulary. We then sort the array and return corresponding indices using `argsort`, and take first `n` elements of the list, which encode positions of closest words in the vocabulary.
"""
logger.info("It is interesting to use Word2Vec to look for synonyms. The following function will return `n` closest words to a given input. To find them, we compute the norm of $|w_i - v|$, where $v$ is the vector corresponding to our input word, and $w_i$ is the encoding of $i$-th word in the vocabulary. We then sort the array and return corresponding indices using `argsort`, and take first `n` elements of the list, which encode positions of closest words in the vocabulary.")

def close_words(x, n = 5):
  vec = embedder(torch.tensor(vocab[x]))
  top5 = np.linalg.norm(vectors.detach().numpy() - vec.detach().numpy(), axis = 1).argsort()[:n]
  return [ vocab.itos[x] for x in top5 ]

close_words('microsoft')

close_words('basketball')

close_words('funds')

"""
## Takeaway

Using clever techniques such as CBoW, we can train Word2Vec model. You may also try to train skip-gram model that is trained to predict the neighboring word given the central one, and see how well it performs.
"""
logger.info("## Takeaway")

logger.info("\n\n[DONE]", bright=True)