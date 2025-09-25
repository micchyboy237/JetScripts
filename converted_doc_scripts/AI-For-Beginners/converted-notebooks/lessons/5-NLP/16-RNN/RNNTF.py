from jet.logger import logger
from tensorflow import keras
import numpy as np
import os
import shutil
import sys
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
# Recurrent neural networks

In the previous module, we covered rich semantic representations of text. The architecture we've been using captures the aggregated meaning of words in a sentence, but it does not take into account the **order** of the words, because the aggregation operation that follows the embeddings removes this information from the original text. Because these models are unable to represent word ordering, they cannot solve more complex or ambiguous tasks such as text generation or question answering.

To capture the meaning of a text sequence, we'll use a neural network architecture called **recurrent neural network**, or RNN. When using an RNN, we pass our sentence through the network one token at a time, and the network produces some **state**, which we then pass to the network again with the next token.

![Image showing an example recurrent neural network generation.](images/rnn.png)

Given the input sequence of tokens $X_0,\dots,X_n$, the RNN creates a sequence of neural network blocks, and trains this sequence end-to-end using backpropagation. Each network block takes a pair $(X_i,S_i)$ as an input, and produces $S_{i+1}$ as a result. The final state $S_n$ or output $Y_n$ goes into a linear classifier to produce the result. All network blocks share the same weights, and are trained end-to-end using one back propagation pass.

> The figure above shows recurrent neural network in the unrolled form (on the left), and in more compact recurrent representation (on the right). It is important to realize that all RNN Cells have the same **shareable weights**.

Because state vectors $S_0,\dots,S_n$ are passed through the network, the RNN is able to learn sequential dependencies between words. For example, when the word *not* appears somewhere in the sequence, it can learn to negate certain elements within the state vector.

Inside, each RNN cell contains two weight matrices: $W_H$ and $W_I$, and bias $b$. At each RNN step, given input $X_i$ and input state $S_i$, output state is calculated as $S_{i+1} = f(W_H\times S_i + W_I\times X_i+b)$, where $f$ is an activation function (often $\tanh$).

> For problems like text generation (that we will cover in the next unit) or machine translation we also want to get some output value at each RNN step. In this case, there is also another matrix $W_O$, and output is caluclated as $Y_i=f(W_O\times S_i+b_O)$.

Let's see how recurrent neural networks can help us classify our news dataset.

> For the sandbox environment, we need to run the following cell to make sure the required library is installed, and data is prefetched. If you are running locally, you can skip the following cell.
"""
logger.info("# Recurrent neural networks")

# !{sys.executable} -m pip install --quiet tensorflow_datasets==4.4.0
# !cd ~ && wget -q -O - https://mslearntensorflowlp.blob.core.windows.net/data/tfds-ag-news.tgz | tar xz


physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices)>0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

ds_train, ds_test = tfds.load('ag_news_subset').values()

"""
When training large models, GPU memory allocation may become a problem. We also may need to experiment with different minibatch sizes, so that the data fits into our GPU memory, yet the training is fast enough. If you are running this code on your own GPU machine, you may experiment with adjusting minibatch size to speed up training.

> **Note**: Certain versions of NVidia drivers are known not to release the memory after training the model. We are running several examples in this notebooks, and it might cause memory to be exhausted in certain setups, especially if you are doing your own experiments as part of the same notebook. If you encounter some weird errors when starting to train the model, you may want to restart notebook kernel.
"""
logger.info("When training large models, GPU memory allocation may become a problem. We also may need to experiment with different minibatch sizes, so that the data fits into our GPU memory, yet the training is fast enough. If you are running this code on your own GPU machine, you may experiment with adjusting minibatch size to speed up training.")

batch_size = 16
embed_size = 64

"""
## Simple RNN classifier

In the case of a simple RNN, each recurrent unit is a simple linear network, which takes in an input vector and state vector, and produces a new state vector. In Keras, this can be represented by the `SimpleRNN` layer.

While we can pass one-hot encoded tokens to the RNN layer directly, this is not a good idea because of their high dimensionality. Therefore, we will use an embedding layer to lower the dimensionality of word vectors, followed by an RNN layer, and finally a `Dense` classifier.

> **Note**: In cases where the dimensionality isn't so high, for example when using character-level tokenization, it might make sense to pass one-hot encoded tokens directly into the RNN cell.
"""
logger.info("## Simple RNN classifier")

vocab_size = 20000

vectorizer = keras.layers.experimental.preprocessing.TextVectorization(
    max_tokens=vocab_size,
    input_shape=(1,))

model = keras.models.Sequential([
    vectorizer,
    keras.layers.Embedding(vocab_size, embed_size),
    keras.layers.SimpleRNN(16),
    keras.layers.Dense(4,activation='softmax')
])

model.summary()

"""
> **Note:** We use an untrained embedding layer here for simplicity, but for better results we can use a pretrained embedding layer using Word2Vec, as described in the previous unit. It would be a good exercise for you to adapt this code to work with pretrained embeddings.

Now let's train our RNN. RNNs in general are quite difficult to train, because once the RNN cells are unrolled along the sequence length, the resulting number of layers involved in backpropagation is quite large. Thus we need to select a smaller learning rate, and train the network on a larger dataset to produce good results. This can take quite a long time, so using a GPU is preferred.

To speed things up, we will only train the RNN model on news titles, omitting the description. You can try training with description and see if you can get the model to train.
"""
logger.info("Now let's train our RNN. RNNs in general are quite difficult to train, because once the RNN cells are unrolled along the sequence length, the resulting number of layers involved in backpropagation is quite large. Thus we need to select a smaller learning rate, and train the network on a larger dataset to produce good results. This can take quite a long time, so using a GPU is preferred.")

def extract_title(x):
    return x['title']

def tupelize_title(x):
    return (extract_title(x),x['label'])

logger.debug('Training vectorizer')
vectorizer.adapt(ds_train.take(2000).map(extract_title))

model.compile(loss='sparse_categorical_crossentropy',metrics=['acc'], optimizer='adam')
model.fit(ds_train.map(tupelize_title).batch(batch_size),validation_data=ds_test.map(tupelize_title).batch(batch_size))

"""
> **Note** that accuracy is likely to be lower here, because we are training only on news titles.

## Revisiting variable sequences 

Remember that the `TextVectorization` layer will automatically pad sequences of variable length in a minibatch with pad tokens. It turns out that those tokens also take part in training, and they can complicate convergence of the model.

There are several approaches we can take to minimize the amount of padding. One of them is to reorder the dataset by sequence length and group all sequences by size. This can be done using the `tf.data.experimental.bucket_by_sequence_length` function (see [documentation](https://www.tensorflow.org/api_docs/python/tf/data/experimental/bucket_by_sequence_length)). 

Another approach is to use **masking**. In Keras, some layers support additional input that shows which tokens should be taken into account when training. To incorporate masking into our model, we can either include a separate `Masking` layer ([docs](https://keras.io/api/layers/core_layers/masking/)), or we can specify the `mask_zero=True` parameter of our `Embedding` layer.

> **Note**: This training will take around 5 minutes to complete one epoch on the whole dataset. Feel free to interrupt training at any time if you run out of patience. What you can also do is limit the amount of data used for training, by adding `.take(...)` clause after `ds_train` and `ds_test` datasets.
"""
logger.info("## Revisiting variable sequences")

def extract_text(x):
    return x['title']+' '+x['description']

def tupelize(x):
    return (extract_text(x),x['label'])

model = keras.models.Sequential([
    vectorizer,
    keras.layers.Embedding(vocab_size,embed_size,mask_zero=True),
    keras.layers.SimpleRNN(16),
    keras.layers.Dense(4,activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',metrics=['acc'], optimizer='adam')
model.fit(ds_train.map(tupelize).batch(batch_size),validation_data=ds_test.map(tupelize).batch(batch_size))

"""
Now that we're using masking, we can train the model on the whole dataset of titles and descriptions.

> **Note**: Have you noticed that we have been using vectorizer trained on the news titles, and not the whole body of the article? Potentially, this can cause some of the the tokens to be ignored, so it is better to re-train the vectorizer. However, it might only have very small effect, so we will stick to the previous pre-trained vectorizer for the sake of simplicity.

## LSTM: Long short-term memory

One of the main problems of RNNs is **vanishing gradients**. RNNs can be pretty long, and may have a hard time propagating the gradients all the way back to the first layer of the network during backpropagation. When this happens, the network cannot learn relationships between distant tokens. One way to avoid this problem is to introduce **explicit state management** by using **gates**. The two most common architectures that introduce gates are **long short-term memory** (LSTM) and **gated relay unit** (GRU). We'll cover LSTMs here.

![Image showing an example long short term memory cell](images/long-short-term-memory-cell.svg)

An LSTM network is organized in a manner similar to an RNN, but there are two states that are passed from layer to layer: the actual state $c$, and the hidden vector $h$. At each unit, the hidden vector $h_{t-1}$ is combined with input $x_t$, and together they control what happens to the state $c_t$ and output $h_{t}$ through **gates**. Each gate has sigmoid activation (output in the range $[0,1]$), which can be thought of as a bitwise mask when multiplied by the state vector. LSTMs have the following gates (from left to right on the picture above):
* **forget gate** which determines which components of the vector $c_{t-1}$ we need to forget, and which to pass through. 
* **input gate** which determines how much information from the input vector and previous hidden vector should be incorporated into the state vector.
* **output gate** which takes the new state vector and decides which of its components will be used to produce the new hidden vector $h_t$.

The components of the state $c$ can be thought of as flags that can be switched on and off. For example, when we encounter the name *Alice* in the sequence, we guess that it refers to a woman, and raise the flag in the state that says we have a female noun in the sentence. When we further encounter the words *and Tom*, we will raise the flag that says we have a plural noun. Thus by manipulating state we can keep track of the grammatical properties of the sentence.

> **Note**: Here's a great resource for understanding the internals of LSTMs: [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) by Christopher Olah.

While the internal structure of an LSTM cell may look complex, Keras hides this implementation inside the `LSTM` layer, so the only thing we need to do in the example above is to replace the recurrent layer:
"""
logger.info("## LSTM: Long short-term memory")

model = keras.models.Sequential([
    vectorizer,
    keras.layers.Embedding(vocab_size, embed_size),
    keras.layers.LSTM(8),
    keras.layers.Dense(4,activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',metrics=['acc'], optimizer='adam')
model.fit(ds_train.map(tupelize).batch(8),validation_data=ds_test.map(tupelize).batch(8))

"""
> **Note** that training LSTMs is also quite slow, and you may not seem much increase in accuracy in the beginning of training. You may need to continue training for some time to achieve good accuracy.

## Bidirectional and multilayer RNNs

In our examples so far, the recurrent networks operate from the beginning of a sequence until the end. This feels natural to us because it follows the same direction in which we read or listen to speech. However, for scenarios which require random access of the input sequence, it makes more sense to run the recurrent computation in both directions. RNNs that allow computations in both directions are called **bidirectional** RNNs, and they can be created by wrapping the recurrent layer with a special `Bidirectonal` layer.

> **Note**: The `Bidirectional` layer makes two copies of the layer within it, and sets the `go_backwards` property of one of those copies to `True`, making it go in the opposite direction along the sequence.

Recurrent networks, unidirectional or bidirectional, capture patterns within a sequence, and store them into state vectors or return them as output. As with convolutional networks, we can build another recurrent layer following the first one to capture higher level patterns, built from lower level patterns extracted by the first layer. This leads us to the notion of a **multi-layer RNN**, which consists of two or more recurrent networks, where the output of the previous layer is passed to the next layer as input.

![Image showing a Multilayer long-short-term-memory- RNN](images/multi-layer-lstm.jpg)

*Picture from [this wonderful post](https://towardsdatascience.com/from-a-lstm-cell-to-a-multilayer-lstm-network-with-pytorch-2899eb5696f3) by Fernando López.*

Keras makes constructing these networks an easy task, because you just need to add more recurrent layers to the model. For all layers except the last one, we need to specify `return_sequences=True` parameter, because we need the layer to return all intermediate states, and not just the final state of the recurrent computation.

Let's build a two-layer bidirectional LSTM for our classification problem.

> **Note** this code again takes quite a long time to complete, but it gives us highest accuracy we have seen so far. So maybe it is worth waiting and seeing the result.
"""
logger.info("## Bidirectional and multilayer RNNs")

model = keras.models.Sequential([
    vectorizer,
    keras.layers.Embedding(vocab_size, 128, mask_zero=True),
    keras.layers.Bidirectional(keras.layers.LSTM(64,return_sequences=True)),
    keras.layers.Bidirectional(keras.layers.LSTM(64)),
    keras.layers.Dense(4,activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',metrics=['acc'], optimizer='adam')
model.fit(ds_train.map(tupelize).batch(batch_size),
          validation_data=ds_test.map(tupelize).batch(batch_size))

"""
## RNNs for other tasks

Up until now, we've focused on using RNNs to classify sequences of text. But they can handle many more tasks, such as text generation and machine translation &mdash; we'll consider those tasks in the next unit.
"""
logger.info("## RNNs for other tasks")

logger.info("\n\n[DONE]", bright=True)