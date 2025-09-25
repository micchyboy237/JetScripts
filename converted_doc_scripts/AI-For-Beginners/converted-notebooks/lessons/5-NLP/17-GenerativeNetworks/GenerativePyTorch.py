from jet.logger import logger
from torchnlp import *
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
# Generative networks

Recurrent Neural Networks (RNNs) and their gated cell variants such as Long Short Term Memory Cells (LSTMs) and Gated Recurrent Units (GRUs) provided a mechanism for language modeling, i.e. they can learn word ordering and provide predictions for next word in a sequence. This allows us to use RNNs for **generative tasks**, such as ordinary text generation, machine translation, and even image captioning.

In RNN architecture we discussed in the previous unit, each RNN unit produced next hidden state as an output. However, we can also add another output to each recurrent unit, which would allow us to output a **sequence** (which is equal in length to the original sequence). Moreover, we can use RNN units that do not accept an input at each step, and just take some initial state vector, and then produce a sequence of outputs.

In this notebook, we will focus on simple generative models that help us generate text. For simplicity, let's build **character-level network**, which generates text letter by letter. During training, we need to take some text corpus, and split it into letter sequences.
"""
logger.info("# Generative networks")

train_dataset,test_dataset,classes,vocab = load_dataset()

"""
## Building character vocabulary

To build character-level generative network, we need to split text into individual characters instead of words. This can be done by defining a different tokenizer:
"""
logger.info("## Building character vocabulary")

def char_tokenizer(words):
    return list(words) #[word for word in words]

counter = collections.Counter()
for (label, line) in train_dataset:
    counter.update(char_tokenizer(line))
vocab = torchtext.vocab.vocab(counter)

vocab_size = len(vocab)
logger.debug(f"Vocabulary size = {vocab_size}")
logger.debug(f"Encoding of 'a' is {vocab.get_stoi()['a']}")
logger.debug(f"Character with code 13 is {vocab.get_itos()[13]}")

"""
Let's see the example of how we can encode the text from our dataset:
"""
logger.info("Let's see the example of how we can encode the text from our dataset:")

def enc(x):
    return torch.LongTensor(encode(x,voc=vocab,tokenizer=char_tokenizer))

enc(train_dataset[0][1])

"""
## Training a generative RNN

The way we will train RNN to generate text is the following. On each step, we will take a sequence of characters of length `nchars`, and ask the network to generate next output character for each input character:

![Image showing an example RNN generation of the word 'HELLO'.](images/rnn-generate.png)

Depending on the actual scenario, we may also want to include some special characters, such as *end-of-sequence* `<eos>`. In our case, we just want to train the network for endless text generation, thus we will fix the size of each sequence to be equal to `nchars` tokens. Consequently, each training example will consist of `nchars` inputs and `nchars` outputs (which are input sequence shifted one symbol to the left). Minibatch will consist of several such sequences.

The way we will generate minibatches is to take each news text of length `l`, and generate all possible input-output combinations from it (there will be `l-nchars` such combinations). They will form one minibatch, and size of minibatches would be different at each training step.
"""
logger.info("## Training a generative RNN")

nchars = 100

def get_batch(s,nchars=nchars):
    ins = torch.zeros(len(s)-nchars,nchars,dtype=torch.long,device=device)
    outs = torch.zeros(len(s)-nchars,nchars,dtype=torch.long,device=device)
    for i in range(len(s)-nchars):
        ins[i] = enc(s[i:i+nchars])
        outs[i] = enc(s[i+1:i+nchars+1])
    return ins,outs

get_batch(train_dataset[0][1])

"""
Now let's define generator network. It can be based on any recurrent cell which we discussed in the previous unit (simple, LSTM or GRU). In our example we will use LSTM.

Because the network takes characters as input, and vocabulary size is pretty small, we do not need embedding layer, one-hot-encoded input can directly go to LSTM cell. However, because we pass character numbers as input, we need to one-hot-encode them before passing to LSTM. This is done by calling `one_hot` function during `forward` pass. Output encoder would be a linear layer that will convert hidden state into one-hot-encoded output.
"""
logger.info("Now let's define generator network. It can be based on any recurrent cell which we discussed in the previous unit (simple, LSTM or GRU). In our example we will use LSTM.")

class LSTMGenerator(torch.nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.rnn = torch.nn.LSTM(vocab_size,hidden_dim,batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, s=None):
        x = torch.nn.functional.one_hot(x,vocab_size).to(torch.float32)
        x,s = self.rnn(x,s)
        return self.fc(x),s

"""
During training, we want to be able to sample generated text. To do that, we will define `generate` function that will produce output string of length `size`, starting from the initial string `start`.

The way it works is the following. First, we will pass the whole start string through the network, and take output state `s` and next predicted character `out`. Since `out` is one-hot encoded, we take `argmax` to get the index of the character `nc` in the vocabulary, and use `itos` to figure out the actual character and append it to the resulting list of characters `chars`. This process of generating one character is repeated `size` times to generate required number of characters.
"""
logger.info("During training, we want to be able to sample generated text. To do that, we will define `generate` function that will produce output string of length `size`, starting from the initial string `start`.")

def generate(net,size=100,start='today '):
        chars = list(start)
        out, s = net(enc(chars).view(1,-1).to(device))
        for i in range(size):
            nc = torch.argmax(out[0][-1])
            chars.append(vocab.get_itos()[nc])
            out, s = net(nc.view(1,-1),s)
        return ''.join(chars)

"""
Now let's do the training! Training loop is almost the same as in all our previous examples, but instead of accuracy we print sampled generated text every 1000 epochs.

Special attention needs to be paid to the way we compute loss. We need to compute loss given one-hot-encoded output `out`, and expected text `text_out`, which is the list of character indices. Luckily, the `cross_entropy` function expects unnormalized network output as first argument, and class number as the second, which is exactly what we have. It also performs automatic averaging over minibatch size.

We also limit the training by `samples_to_train` samples, in order not to wait for too long. We encourage you to experiment and try longer training, possibly for several epochs (in which case you would need to create another loop around this code).
"""
logger.info("Now let's do the training! Training loop is almost the same as in all our previous examples, but instead of accuracy we print sampled generated text every 1000 epochs.")

net = LSTMGenerator(vocab_size,64).to(device)

samples_to_train = 10000
optimizer = torch.optim.Adam(net.parameters(),0.01)
loss_fn = torch.nn.CrossEntropyLoss()
net.train()
for i,x in enumerate(train_dataset):
    if len(x[1])-nchars<10:
        continue
    samples_to_train-=1
    if not samples_to_train: break
    text_in, text_out = get_batch(x[1])
    optimizer.zero_grad()
    out,s = net(text_in)
    loss = torch.nn.functional.cross_entropy(out.view(-1,vocab_size),text_out.flatten()) #cross_entropy(out,labels)
    loss.backward()
    optimizer.step()
    if i%1000==0:
        logger.debug(f"Current loss = {loss.item()}")
        logger.debug(generate(net))

"""
This example already generates some pretty good text, but it can be further improved in several ways:
* **Better minibatch generation**. The way we prepared data for training was to generate one minibatch from one sample. This is not ideal, because minibatches are all of different sizes, and some of them even cannot be generated, because the text is smaller than `nchars`. Also, small minibatches do not load GPU sufficiently enough. It would be wiser to get one large chunk of text from all samples, then generate all input-output pairs, shuffle them, and generate minibatches of equal size.
* **Multilayer LSTM**. It makes sense to try 2 or 3 layers of LSTM cells. As we mentioned in the previous unit, each layer of LSTM extracts certain patterns from text, and in case of character-level generator we can expect lower LSTM level to be responsible for extracting syllables, and higher levels - for words and word combinations. This can be simply implemented by passing number-of-layers parameter to LSTM constructor.
* You may also want to experiment with **GRU units** and see which ones perform better, and with **different hidden layer sizes**. Too large hidden layer may result in overfitting (e.g. network will learn exact text), and smaller size might not produce good result.

## Soft text generation and temperature

In the previous definition of `generate`, we were always taking the character with highest probability as the next character in generated text. This resulted in the fact that the text often "cycled" between the same character sequences again and again, like in this example:
```
today of the second the company and a second the company ...
```

However, if we look at the probability distribution for the next character, it could be that the difference between a few highest probabilities is not huge, e.g. one character can have probability 0.2, another - 0.19, etc. For example, when looking for the next character in the sequence '*play*', next character can equally well be either space, or **e** (as in the word *player*).

This leads us to the conclusion that it is not always "fair" to select the character with higher probability, because choosing the second highest might still lead us to meaningful text. It is more wise to **sample** characters from the probability distribution given by the network output.

This sampling can be done using `multinomial` function that implements so-called **multinomial distribution**. A function that implements this **soft** text generation is defined below:
"""
logger.info("## Soft text generation and temperature")

def generate_soft(net,size=100,start='today ',temperature=1.0):
        chars = list(start)
        out, s = net(enc(chars).view(1,-1).to(device))
        for i in range(size):
            out_dist = out[0][-1].div(temperature).exp()
            nc = torch.multinomial(out_dist,1)[0]
            chars.append(vocab.get_itos()[nc])
            out, s = net(nc.view(1,-1),s)
        return ''.join(chars)

for i in [0.3,0.8,1.0,1.3,1.8]:
    logger.debug(f"--- Temperature = {i}\n{generate_soft(net,size=300,start='Today ',temperature=i)}\n")

"""
We have introduced one more parameter called **temperature**, which is used to indicate how hard we should stick to the highest probability. If temperature is 1.0, we do fair multinomial sampling, and when temperature goes to infinity - all probabilities become equal, and we randomly select next character. In the example below we can observe that the text becomes meaningless when we increase the temperature too much, and it resembles "cycled" hard-generated text when it becomes closer to 0.
"""
logger.info("We have introduced one more parameter called **temperature**, which is used to indicate how hard we should stick to the highest probability. If temperature is 1.0, we do fair multinomial sampling, and when temperature goes to infinity - all probabilities become equal, and we randomly select next character. In the example below we can observe that the text becomes meaningless when we increase the temperature too much, and it resembles "cycled" hard-generated text when it becomes closer to 0.")

logger.info("\n\n[DONE]", bright=True)