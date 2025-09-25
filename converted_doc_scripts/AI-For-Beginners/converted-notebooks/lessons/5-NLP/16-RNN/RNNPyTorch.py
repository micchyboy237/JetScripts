from jet.logger import logger
from torchnlp import *
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
# Recurrent neural networks

In the previous module, we have been using rich semantic representations of text, and a simple linear classifier on top of the embeddings. What this architecture does is to capture aggregated meaning of words in a sentence, but it does not take into account the **order** of words, because aggregation operation on top of embeddings removed this information from the original text. Because these models are unable to model word ordering, they cannot solve more complex or ambiguous tasks such as text generation or question answering.

To capture the meaning of text sequence, we need to use another neural network architecture, which is called a **recurrent neural network**, or RNN. In RNN, we pass our sentence through the network one symbol at a time, and the network produces some **state**, which we then pass to the network again with the next symbol.

<img alt="RNN" src="images/rnn.png" width="60%"/>

Given the input sequence of tokens $X_0,\dots,X_n$, RNN creates a sequence of neural network blocks, and trains this sequence end-to-end using back propagation. Each network block takes a pair $(X_i,S_i)$ as an input, and produces $S_{i+1}$ as a result. Final state $S_n$ or output $X_n$ goes into a linear classifier to produce the result. All network blocks share the same weights, and are trained end-to-end using one back propagation pass.

Because state vectors $S_0,\dots,S_n$ are passed through the network, it is able to learn the sequential dependencies between words. For example, when the word *not* appears somewhere in the sequence, it can learn to negate certain elements within the state vector, resulting in negation.  

> Since weights of all RNN blocks on the picture are shared, the same picture can be represented as one block (on the right) with a recurrent feedback loop, which passes output state of the network back to the input.

Let's see how recurrent neural networks can help us classify our news dataset.
"""
logger.info("# Recurrent neural networks")

train_dataset, test_dataset, classes, vocab = load_dataset()
vocab_size = len(vocab)

"""
## Simple RNN classifier

In case of simple RNN, each recurrent unit is a simple linear network, which takes concatenated input vector and state vector, and produce a new state vector. PyTorch represents this unit with `RNNCell` class, and a networks of such cells - as `RNN` layer.

To define an RNN classifier, we will first apply an embedding layer to lower the dimensionality of input vocabulary, and then have RNN layer on top of it:
"""
logger.info("## Simple RNN classifier")

class RNNClassifier(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_class):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.rnn = torch.nn.RNN(embed_dim,hidden_dim,batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, num_class)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.embedding(x)
        x,h = self.rnn(x)
        return self.fc(x.mean(dim=1))

"""
> **Note:** We use untrained embedding layer here for simplicity, but for even better results we can use pre-trained embedding layer with Word2Vec or GloVe embeddings, as described in the previous unit. For better understanding, you might want to adapt this code to work with pre-trained embeddings.

In our case, we will use padded data loader, so each batch will have a number of padded sequences of the same length. RNN layer will take the sequence of embedding tensors, and produce two outputs: 
* $x$ is a sequence of RNN cell outputs at each step
* $h$ is a final hidden state for the last element of the sequence

We then apply a fully-connected linear classifier to get the number of class.

> **Note:** RNNs are quite difficult to train, because once the RNN cells are unrolled along the sequence length, the resulting number of layers involved in back propagation is quite large. Thus we need to select small learning rate, and train the network on larger dataset to produce good results. It can take quite a long time, so using GPU is preferred.
"""
logger.info("In our case, we will use padded data loader, so each batch will have a number of padded sequences of the same length. RNN layer will take the sequence of embedding tensors, and produce two outputs:")

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, collate_fn=padify, shuffle=True)
net = RNNClassifier(vocab_size,64,32,len(classes)).to(device)
train_epoch(net,train_loader, lr=0.001)

"""
## Long Short Term Memory (LSTM)

One of the main problems of classical RNNs is so-called **vanishing gradients** problem. Because RNNs are trained end-to-end in one back-propagation pass, it is having hard times propagating error to the first layers of the network, and thus the network cannot learn relationships between distant tokens. One of the ways to avoid this problem is to introduce **explicit state management** by using so called **gates**. There are two most known architectures of this kind: **Long Short Term Memory** (LSTM) and **Gated Relay Unit** (GRU).

![Image showing an example long short term memory cell](./images/long-short-term-memory-cell.svg)

LSTM Network is organized in a manner similar to RNN, but there are two states that are being passed from layer to layer: actual state $c$, and hidden vector $h$. At each unit, hidden vector $h_i$ is concatenated with input $x_i$, and they control what happens to the state $c$ via **gates**. Each gate is a neural network with sigmoid activation (output in the range $[0,1]$), which can be thought of as bitwise mask when multiplied by the state vector. There are the following gates (from left to right on the picture above):
* **forget gate** takes hidden vector and determines, which components of the vector $c$ we need to forget, and which to pass through. 
* **input gate** takes some information from the input and hidden vector, and inserts it into state.
* **output gate** transforms state via some linear layer with $\tanh$ activation, then selects some of its components using hidden vector $h_i$ to produce new state $c_{i+1}$.

Components of the state $c$ can be thought of as some flags that can be switched on and off. For example, when we encounter a name *Alice* in the sequence, we may want to assume that it refers to female character, and raise the flag in the state that we have female noun in the sentence. When we further encounter phrases *and Tom*, we will raise the flag that we have plural noun. Thus by manipulating state we can supposedly keep track of grammatical properties of sentence parts.

> **Note**: A great resource for understanding internals of LSTM is this great article [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) by Christopher Olah.

While internal structure of LSTM cell may look complex, PyTorch hides this implementation inside `LSTMCell` class, and provides `LSTM` object to represent the whole LSTM layer. Thus, implementation of LSTM classifier will be pretty similar to the simple RNN which we have seen above:
"""
logger.info("## Long Short Term Memory (LSTM)")

class LSTMClassifier(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_class):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.embedding.weight.data = torch.randn_like(self.embedding.weight.data)-0.5
        self.rnn = torch.nn.LSTM(embed_dim,hidden_dim,batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, num_class)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.embedding(x)
        x,(h,c) = self.rnn(x)
        return self.fc(h[-1])

"""
Now let's train our network. Note that training LSTM is also quite slow, and you may not seem much raise in accuracy in the beginning of training. Also, you may need to play with `lr` learning rate parameter to find the learning rate that results in reasonable training speed, and yet does not cause memory waste.
"""
logger.info("Now let's train our network. Note that training LSTM is also quite slow, and you may not seem much raise in accuracy in the beginning of training. Also, you may need to play with `lr` learning rate parameter to find the learning rate that results in reasonable training speed, and yet does not cause memory waste.")

net = LSTMClassifier(vocab_size,64,32,len(classes)).to(device)
train_epoch(net,train_loader, lr=0.001)

"""
## Packed sequences

In our example, we had to pad all sequences in the minibatch with zero vectors. While it results in some memory waste, with RNNs it is more critical that additional RNN cells are created for the padded input items, which take part in training, yet do not carry any important input information. It would be much better to train RNN only to the actual sequence size.

To do that, a special format of padded sequence storage is introduced in PyTorch. Suppose we have input padded minibatch which looks like this:
```
[[1,2,3,4,5],
 [6,7,8,0,0],
 [9,0,0,0,0]]
```
Here 0 represents padded values, and the actual length vector of input sequences is `[5,3,1]`.

In order to effectively train RNN with padded sequence, we want to begin training first group of RNN cells with large minibatch (`[1,6,9]`), but then end processing of third sequence, and continue training with shorted minibatches (`[2,7]`, `[3,8]`), and so on. Thus, packed sequence is represented as one vector - in our case `[1,6,9,2,7,3,8,4,5]`, and length vector (`[5,3,1]`), from which we can easily reconstruct the original padded minibatch.

To produce packed sequence, we can use `torch.nn.utils.rnn.pack_padded_sequence` function. All recurrent layers, including RNN, LSTM and GRU, support packed sequences as input, and produce packed output, which can be decoded using `torch.nn.utils.rnn.pad_packed_sequence`.

To be able to produce packed sequence, we need to pass length vector to the network, and thus we need a different function to prepare minibatches:
"""
logger.info("## Packed sequences")

def pad_length(b):
    v = [encode(x[1]) for x in b]
    len_seq = list(map(len,v))
    l = max(len_seq)
    return ( # tuple of three tensors - labels, padded features, length sequence
        torch.LongTensor([t[0]-1 for t in b]),
        torch.stack([torch.nn.functional.pad(torch.tensor(t),(0,l-len(t)),mode='constant',value=0) for t in v]),
        torch.tensor(len_seq)
    )

train_loader_len = torch.utils.data.DataLoader(train_dataset, batch_size=16, collate_fn=pad_length, shuffle=True)

"""
Actual network would be very similar to `LSTMClassifier` above, but `forward` pass will receive both padded minibatch and the vector of sequence lengths. After computing the embedding, we compute packed sequence, pass it to LSTM layer, and then unpack the result back.

> **Note**: We actually do not use unpacked result `x`, because we use output from the hidden layers in the following computations. Thus, we can remove the unpacking altogether from this code. The reason we place it here is for you to be able to modify this code easily, in case you should need to use network output in further computations.
"""
logger.info("Actual network would be very similar to `LSTMClassifier` above, but `forward` pass will receive both padded minibatch and the vector of sequence lengths. After computing the embedding, we compute packed sequence, pass it to LSTM layer, and then unpack the result back.")

class LSTMPackClassifier(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_class):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.embedding.weight.data = torch.randn_like(self.embedding.weight.data)-0.5
        self.rnn = torch.nn.LSTM(embed_dim,hidden_dim,batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, num_class)

    def forward(self, x, lengths):
        batch_size = x.size(0)
        x = self.embedding(x)
        pad_x = torch.nn.utils.rnn.pack_padded_sequence(x,lengths,batch_first=True,enforce_sorted=False)
        pad_x,(h,c) = self.rnn(pad_x)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(pad_x,batch_first=True)
        return self.fc(h[-1])

"""
Now let's do the training:
"""
logger.info("Now let's do the training:")

net = LSTMPackClassifier(vocab_size,64,32,len(classes)).to(device)
train_epoch_emb(net,train_loader_len, lr=0.001,use_pack_sequence=True)

"""
> **Note:** You may have noticed the parameter `use_pack_sequence` that we pass to the training function. Currently, `pack_padded_sequence` function requires length sequence tensor to be on CPU device, and thus training function needs to avoid moving the length sequence data to GPU when training. You can look into implementation of `train_emb` function in the [`torchnlp.py`](torchnlp.py) file.

## Bidirectional and multilayer RNNs

In our examples, all recurrent networks operated in one direction, from beginning of a sequence to the end. It looks natural, because it resembles the way we read and listen to speech. However, since in many practical cases we have random access to the input sequence, it might make sense to run recurrent computation in both directions. Such networks are call **bidirectional** RNNs, and they can be created by passing `bidirectional=True` parameter to RNN/LSTM/GRU constructor.

When dealing with bidirectional network, we would need two hidden state vectors, one for each direction. PyTorch encodes those vectors as one vector of twice larger size, which is quite convenient, because you would normally pass the resulting hidden state to fully-connected linear layer, and you would just need to take this increase in size into account when creating the layer.

Recurrent network, one-directional or bidirectional, captures certain patterns within a sequence, and can store them into state vector or pass into output. As with convolutional networks, we can build another recurrent layer on top of the first one to capture higher level patterns, build from low-level patterns extracted by the first layer. This leads us to the notion of **multi-layer RNN**, which consists of two or more recurrent networks, where output of the previous layer is passed to the next layer as input.

![Image showing a Multilayer long-short-term-memory- RNN](images/multi-layer-lstm.jpg)

*Picture from [this wonderful post](https://towardsdatascience.com/from-a-lstm-cell-to-a-multilayer-lstm-network-with-pytorch-2899eb5696f3) by Fernando López*

PyTorch makes constructing such networks an easy task, because you just need to pass `num_layers` parameter to RNN/LSTM/GRU constructor to build several layers of recurrence automatically. This would also mean that the size of hidden/state vector would increase proportionally, and you would need to take this into account when handling the output of recurrent layers.

## RNNs for other tasks

In this unit, we have seen that RNNs can be used for sequence classification, but in fact, they can handle many more tasks, such as text generation, machine translation, and more. We will consider those tasks in the next unit.
"""
logger.info("## Bidirectional and multilayer RNNs")

logger.info("\n\n[DONE]", bright=True)