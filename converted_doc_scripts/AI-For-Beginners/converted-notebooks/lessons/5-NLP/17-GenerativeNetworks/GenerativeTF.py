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
# Generative networks

Recurrent Neural Networks (RNNs) and their gated cell variants such as Long Short Term Memory Cells (LSTMs) and Gated Recurrent Units (GRUs) provided a mechanism for language modeling, i.e. they can learn word ordering and provide predictions for next word in a sequence. This allows us to use RNNs for **generative tasks**, such as ordinary text generation, machine translation, and even image captioning.

In RNN architecture we discussed in the previous unit, each RNN unit produced next hidden state as an output. However, we can also add another output to each recurrent unit, which would allow us to output a **sequence** (which is equal in length to the original sequence). Moreover, we can use RNN units that do not accept an input at each step, and just take some initial state vector, and then produce a sequence of outputs.

In this notebook, we will focus on simple generative models that help us generate text. For simplicity, let's build **character-level network**, which generates text letter by letter. During training, we need to take some text corpus, and split it into letter sequences.
"""
logger.info("# Generative networks")


ds_train, ds_test = tfds.load('ag_news_subset').values()

"""
## Building character vocabulary

To build character-level generative network, we need to split text into individual characters instead of words. `TextVectorization` layer that we have been using before cannot do that, so we have to options:

* Manually load text and do tokenization 'by hand', as in [this official Keras example](https://keras.io/examples/generative/lstm_character_level_text_generation/)
* Use `Tokenizer` class for character-level tokenization.

We will go with the second option. `Tokenizer` can also be used to tokenize into words, so one should be able to switch from char-level to word-level tokenization quite easily.

To do character-level tokenization, we need to pass `char_level=True` parameter:
"""
logger.info("## Building character vocabulary")

def extract_text(x):
    return x['title']+' '+x['description']

def tupelize(x):
    return (extract_text(x),x['label'])

tokenizer = keras.preprocessing.text.Tokenizer(char_level=True,lower=False)
tokenizer.fit_on_texts([x['title'].numpy().decode('utf-8') for x in ds_train])

"""
We also want to use one special token to denote **end of sequence**, which we will call `<eos>`. Let's add it manually to the vocabulary:
"""
logger.info("We also want to use one special token to denote **end of sequence**, which we will call `<eos>`. Let's add it manually to the vocabulary:")

eos_token = len(tokenizer.word_index)+1
tokenizer.word_index['<eos>'] = eos_token

vocab_size = eos_token + 1

"""
Now, to encode text into sequences of numbers, we can use:
"""
logger.info("Now, to encode text into sequences of numbers, we can use:")

tokenizer.texts_to_sequences(['Hello, world!'])

"""
## Training a generative RNN to generate titles

The way we will train RNN to generate news titles is the following. On each step, we will take one title, which will be fed into an RNN, and for each input character we will ask the network to generate next output character:

![Image showing an example RNN generation of the word 'HELLO'.](./images/rnn-generate.png)

For the last character of our sequence, we will ask the network to generate `<eos>` token.

The main difference between generative RNN that we are using here is that we will take an output from each step of the RNN, and not just from the final cell. This can be achieved by specifying `return_sequences` parameter to the RNN cell.

Thus, during the training, an input to the network would be a sequence of encoded characters of some length, and an output would be a sequence of the same length, but shifted by one element and terminated by `<eos>`. Minibatch will consist of several such sequences, and we would need to use **padding** to align all sequences.

Let's create functions that will transform the dataset for us. Because we want to pad sequences on minibatch level, we will first batch the dataset by calling `.batch()`, and then `map` it in order to do transformation. So, the transformation function will take a whole minibatch as a parameter:
"""
logger.info("## Training a generative RNN to generate titles")

def title_batch(x):
    x = [t.numpy().decode('utf-8') for t in x]
    z = tokenizer.texts_to_sequences(x)
    z = tf.keras.preprocessing.sequence.pad_sequences(z)
    return tf.one_hot(z,vocab_size), tf.one_hot(tf.concat([z[:,1:],tf.constant(eos_token,shape=(len(z),1))],axis=1),vocab_size)

"""
A few important things that we do here:
* We first extract the actual text from the string tensor
* `text_to_sequences` converts the list of strings into a list of integer tensors
* `pad_sequences` pads those tensors to their maximum length
* We finally one-hot encode all the characters, and also do the shifting and `<eos>` appending. We will soon see why we need one-hot-encoded characters

However, this function is **Pythonic**, i.e. it cannot be automatically translated into Tensorflow computational graph. We will get errors if we try to use this function directly in the `Dataset.map` function. We need to enclose this Pythonic call by using `py_function` wrapper:
"""
logger.info("A few important things that we do here:")

def title_batch_fn(x):
    x = x['title']
    a,b = tf.py_function(title_batch,inp=[x],Tout=(tf.float32,tf.float32))
    return a,b

"""
> **Note**: Differentiating between Pythonic and Tensorflow transformation functions may seem a little too complex, and you may be questioning why we do not transform the dataset using standard Python functions before passing it to `fit`. While this definitely can be done, using `Dataset.map` has a huge advantage, because data transformation pipeline is executed using Tensorflow computational graph, which takes advantage of GPU computations, and minimized the need to pass data between CPU/GPU.

Now we can build our generator network and start training. It can be based on any recurrent cell which we discussed in the previous unit (simple, LSTM or GRU). In our example we will use LSTM.

Because the network takes characters as input, and vocabulary size is pretty small, we do not need embedding layer, one-hot-encoded input can directly go into LSTM cell. Output layer would be a `Dense` classifier that will convert LSTM output into one-hot-encoded token numbers.

In addition, since we are dealing with variable-length sequences, we can use `Masking` layer to create a mask that will ignore padded part of the string. This is not strictly needed, because we are not very much interested in everything that goes beyond `<eos>` token, but we will use it for the sake of getting some experience with this layer type. `input_shape` would be `(None, vocab_size)`, where `None` indicates the sequence of variable length, and output shape is `(None,vocab_size)` as well, as you can see from the `summary`:
"""
logger.info("Now we can build our generator network and start training. It can be based on any recurrent cell which we discussed in the previous unit (simple, LSTM or GRU). In our example we will use LSTM.")

model = keras.models.Sequential([
    keras.layers.Masking(input_shape=(None,vocab_size)),
    keras.layers.LSTM(128,return_sequences=True),
    keras.layers.Dense(vocab_size,activation='softmax')
])

model.summary()
model.compile(loss='categorical_crossentropy')

model.fit(ds_train.batch(8).map(title_batch_fn))

"""
## Generating output

Now that we have trained the model, we want to use it to generate some output. First of all, we need a way to decode text represented by a sequence of token numbers. To do this, we could use `tokenizer.sequences_to_texts` function; however, it does not work well with character-level tokenization. Therefore we will take a dictionary of tokens from the tokenizer (called `word_index`), build a reverse map, and write our own decoding function:
"""
logger.info("## Generating output")

reverse_map = {val:key for key, val in tokenizer.word_index.items()}

def decode(x):
    return ''.join([reverse_map[t] for t in x])

"""
Now, let's do generation. We will start with some string `start`, encode it into a sequence `inp`, and then on each step we will call our network to infer the next character. 

Output of the network `out` is a vector of `vocab_size` elements representing probablities of each token, and we can find the most probably token number by using `argmax`. We then append this character to the generated list of tokens, and proceed with generation. This process of generating one character is repeated `size` times to generate required number of characters, and we terminate early when `eos_token` is encountered.
"""
logger.info("Now, let's do generation. We will start with some string `start`, encode it into a sequence `inp`, and then on each step we will call our network to infer the next character.")

def generate(model,size=100,start='Today '):
        inp = tokenizer.texts_to_sequences([start])[0]
        chars = inp
        for i in range(size):
            out = model(tf.expand_dims(tf.one_hot(inp,vocab_size),0))[0][-1]
            nc = tf.argmax(out)
            if nc==eos_token:
                break
            chars.append(nc.numpy())
            inp = inp+[nc]
        return decode(chars)

generate(model)

"""
## Sampling output during training 

Because we do not have any useful metrics such as *accuracy*, the only way we can see that our model is getting better is by **sampling** generated string during training. To do it, we will use **callbacks**, i.e. functions that we can pass to the `fit` function, and that will be called periodically during training.
"""
logger.info("## Sampling output during training")

sampling_callback = keras.callbacks.LambdaCallback(
  on_epoch_end = lambda batch, logs: logger.debug(generate(model))
)

model.fit(ds_train.batch(8).map(title_batch_fn),callbacks=[sampling_callback],epochs=3)

"""
This example already generates some pretty good text, but it can be further improved in several ways:
* **More text**. We have only used titles for our task, but you may want to experiment with full text. Remember that RNNs are not too great with handling long sequences, so it makes sense either to split them into shorted sentences, or to always train on a fixed sequence length of some predefined value `num_chars` (say, 256). You may try to change the example above into such architecture, using [official Keras tutorial](https://keras.io/examples/generative/lstm_character_level_text_generation/) as an inspiration.
* **Multilayer LSTM**. It makes sense to try 2 or 3 layers of LSTM cells. As we mentioned in the previous unit, each layer of LSTM extracts certain patterns from text, and in case of character-level generator we can expect lower LSTM level to be responsible for extracting syllables, and higher levels - for words and word combinations. This can be simply implemented by passing number-of-layers parameter to LSTM constructor.
* You may also want to experiment with **GRU units** and see which ones perform better, and with **different hidden layer sizes**. Too large hidden layer may result in overfitting (e.g. network will learn exact text), and smaller size might not produce good result.

## Soft text generation and temperature

In the previous definition of `generate`, we were always taking the character with highest probability as the next character in generated text. This resulted in the fact that the text often "cycled" between the same character sequences again and again, like in this example:
```
today of the second the company and a second the company ...
```

However, if we look at the probability distribution for the next character, it could be that the difference between a few highest probabilities is not huge, e.g. one character can have probability 0.2, another - 0.19, etc. For example, when looking for the next character in the sequence '*play*', next character can equally well be either space, or **e** (as in the word *player*).

This leads us to the conclusion that it is not always "fair" to select the character with higher probability, because choosing the second highest might still lead us to meaningful text. It is more wise to **sample** characters from the probability distribution given by the network output.

This sampling can be done using `np.multinomial` function that implements so-called **multinomial distribution**. A function that implements this **soft** text generation is defined below:
"""
logger.info("## Soft text generation and temperature")

def generate_soft(model,size=100,start='Today ',temperature=1.0):
        inp = tokenizer.texts_to_sequences([start])[0]
        chars = inp
        for i in range(size):
            out = model(tf.expand_dims(tf.one_hot(inp,vocab_size),0))[0][-1]
            probs = tf.exp(tf.math.log(out)/temperature).numpy().astype(np.float64)
            probs = probs/np.sum(probs)
            nc = np.argmax(np.random.multinomial(1,probs,1))
            if nc==eos_token:
                break
            chars.append(nc)
            inp = inp+[nc]
        return decode(chars)

words = ['Today ','On Sunday ','Moscow, ','President ','Little red riding hood ']

for i in [0.3,0.8,1.0,1.3,1.8]:
    logger.debug(f"\n--- Temperature = {i}")
    for j in range(5):
        logger.debug(generate_soft(model,size=300,start=words[j],temperature=i))

"""
We have introduced one more parameter called **temperature**, which is used to indicate how hard we should stick to the highest probability. If temperature is 1.0, we do fair multinomial sampling, and when temperature goes to infinity - all probabilities become equal, and we randomly select next character. In the example below we can observe that the text becomes meaningless when we increase the temperature too much, and it resembles "cycled" hard-generated text when it becomes closer to 0.
"""
logger.info("We have introduced one more parameter called **temperature**, which is used to indicate how hard we should stick to the highest probability. If temperature is 1.0, we do fair multinomial sampling, and when temperature goes to infinity - all probabilities become equal, and we randomly select next character. In the example below we can observe that the text becomes meaningless when we increase the temperature too much, and it resembles "cycled" hard-generated text when it becomes closer to 0.")

logger.info("\n\n[DONE]", bright=True)