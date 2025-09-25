from jet.logger import logger
from tensorflow import keras
import numpy as np
import os
import shutil
import tensorflow as tf
import tensorflow_datasets as tfds


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


"""
We will start by loading the dateset:
"""
logger.info("We will start by loading the dateset:")

ds_train, ds_test = tfds.load('ag_news_subset').values()

"""
## CBoW Model

CBoW learns to predict a word based on the $2N$ neighboring words. For example, when $N=1$, we will get the following pairs from the sentence *I like to train networks*: (like,I), (I, like), (to, like), (like,to), (train,to), (to, train), (networks, train), (train,networks). Here, first word is the neighboring word used as an input, and second word is the one we are predicting.

To build a network to predict next word, we will need to supply neighboring word as input, and get word number as output. The architecture of CBoW network is the following:

* Input word is passed through the embedding layer. This very embedding layer would be our Word2Vec embedding, thus we will define it separately as `embedder` variable. We will use embedding size = 30 in this example, even though you might want to experiment with higher dimensions (real word2vec has 300)
* Embedding vector would then be passed to a dense layer that will predict output word. Thus it has the `vocab_size` neurons.

Embedding layer in Keras automatically knows how to convert numeric input into one-hot encoding, so that we do not have to one-hot-encode input word separately. We specify `input_length=1` to indicate that we want just one word in the input sequence - normally embedding layer is designed to work with longer sequences.

For the output, if we use `sparse_categorical_crossentropy` as loss function, we would also have to provide just word numbers as expected results, without one-hot encoding.

We will set `vocab_size` to 5000 to limit computations a bit. We will also define a vectorizer which we will use later.
"""
logger.info("## CBoW Model")

vocab_size = 5000

vectorizer = keras.layers.experimental.preprocessing.TextVectorization(max_tokens=vocab_size,input_shape=(1,))
embedder = keras.layers.Embedding(vocab_size,30,input_length=1)

model = keras.Sequential([
    embedder,
    keras.layers.Dense(vocab_size,activation='softmax')
])

model.summary()

"""
Let's initialize the vectorizer and get out the vocabulary:
"""
logger.info("Let's initialize the vectorizer and get out the vocabulary:")

def extract_text(x):
    return x['title']+' '+x['description']

vectorizer.adapt(ds_train.take(500).map(extract_text))
vocab = vectorizer.get_vocabulary()

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
logger.debug(to_cbow(vectorizer('I like to train networks')))

"""
Let's prepare the training dataset. We will go through all news, call `to_cbow` to get the list of word pairs, and add those pairs to `X` and `Y`. For the sake of time, we will only consider first 10k news items - you can easily remove the limitation in case you have more time to wait, and want to get better embeddings :)
"""
logger.info("Let's prepare the training dataset. We will go through all news, call `to_cbow` to get the list of word pairs, and add those pairs to `X` and `Y`. For the sake of time, we will only consider first 10k news items - you can easily remove the limitation in case you have more time to wait, and want to get better embeddings :)")

X = []
Y = []
for i,x in zip(range(10000),ds_train.map(extract_text).as_numpy_iterator()):
    for w1, w2 in to_cbow(vectorizer(x),window_size=1):
        X.append(tf.expand_dims(w1,0))
        Y.append(tf.expand_dims(w2,0))

"""
We will also convert that data to one dataset, and batch it for training:
"""
logger.info("We will also convert that data to one dataset, and batch it for training:")

ds = tf.data.Dataset.from_tensor_slices((X,Y)).batch(256)

"""
Now let's do the actual training. We will use `SGD` optimizer with pretty high learning rate. You can also try playing around with other optimizers, such as `Adam`. We will train for 200 epochs to begin with - and you can re-run this cell if you want even lower loss.
"""
logger.info("Now let's do the actual training. We will use `SGD` optimizer with pretty high learning rate. You can also try playing around with other optimizers, such as `Adam`. We will train for 200 epochs to begin with - and you can re-run this cell if you want even lower loss.")

model.compile(optimizer=keras.optimizers.SGD(lr=0.1),loss='sparse_categorical_crossentropy')
model.fit(ds,epochs=200)

"""
## Trying out Word2Vec

To use Word2Vec, let's extract vectors corresponding to all words in our vocabulary:
"""
logger.info("## Trying out Word2Vec")

vectors = embedder(vectorizer(vocab))
vectors = tf.reshape(vectors,(-1,30)) # we need reshape to get rid of extra dimension

"""
Let's see, for example, how the word **Paris** is encoded into a vector:
"""
logger.info("Let's see, for example, how the word **Paris** is encoded into a vector:")

paris_vec = embedder(vectorizer('paris'))[0]
logger.debug(paris_vec)

"""
It is interesting to use Word2Vec to look for synonyms. The following function will return `n` closest words to a given input. To find them, we compute the norm of $|w_i - v|$, where $v$ is the vector corresponding to our input word, and $w_i$ is the encoding of $i$-th word in the vocabulary. We then sort the array and return corresponding indices using `argsort`, and take first `n` elements of the list, which encode positions of closest words in the vocabulary.
"""
logger.info("It is interesting to use Word2Vec to look for synonyms. The following function will return `n` closest words to a given input. To find them, we compute the norm of $|w_i - v|$, where $v$ is the vector corresponding to our input word, and $w_i$ is the encoding of $i$-th word in the vocabulary. We then sort the array and return corresponding indices using `argsort`, and take first `n` elements of the list, which encode positions of closest words in the vocabulary.")

def close_words(x,n=5):
  vec = embedder(vectorizer(x))[0]
  top5 = np.linalg.norm(vectors-vec,axis=1).argsort()[:n]
  return [ vocab[x] for x in top5 ]

close_words('paris')

close_words('china')

close_words('official')

"""
## Takeaway

Using clever techniques such as CBoW, we can train Word2Vec model. You may also try to train skip-gram model that is trained to predict the neighboring word given the central one, and see how well it performs.
"""
logger.info("## Takeaway")

logger.info("\n\n[DONE]", bright=True)