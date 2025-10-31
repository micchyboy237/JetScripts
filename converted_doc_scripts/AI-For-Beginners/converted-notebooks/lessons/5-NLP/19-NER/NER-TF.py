from jet.logger import logger
from tensorflow import keras
import numpy as np
import os
import pandas as pd
import shutil


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
# Named Entity Recognition (NER)

This notebook is from [AI for Beginners Curriculum](http://aka.ms/ai-beginners).

In this example, we will learn how to train NER model on [Annotated Corpus for Named Entity Recognition](https://www.kaggle.com/datasets/abhinavwalia95/entity-annotated-corpus) Dataset from Kaggle. Before procedding, please donwload [ner_dataset.csv](https://www.kaggle.com/datasets/abhinavwalia95/entity-annotated-corpus?resource=download&select=ner_dataset.csv) file into current directory.
"""
logger.info("# Named Entity Recognition (NER)")


"""
## Preparing the Dataset 

We will start by reading the dataset into a dataframe. If you want to learn more about using Pandas, visit a [lesson on data processing](https://github.com/microsoft/Data-Science-For-Beginners/tree/main/2-Working-With-Data/07-python) in our [Data Science for Beginners](http://aka.ms/datascience-beginners)
"""
logger.info("## Preparing the Dataset")

csv_file = "/Users/jethroestrada/Desktop/External_Projects/Jet_Projects/JetScripts/data/kaggle/ner_dataset.csv"
df = pd.read_csv(csv_file,encoding='unicode-escape')
df.head()

"""
Let's get unique tags and create lookup dictionaries that we can use to convert tags into class numbers:
"""
logger.info("Let's get unique tags and create lookup dictionaries that we can use to convert tags into class numbers:")

tags = df.Tag.unique()
tags

id2tag = dict(enumerate(tags))
tag2id = { v : k for k,v in id2tag.items() }

id2tag[0]

"""
Now we need to do the same with vocabulary. For simplicity, we will create vocabulary without taking word frequency into account; in real life you might want to use Keras vectorizer, and limit the number of words.
"""
logger.info("Now we need to do the same with vocabulary. For simplicity, we will create vocabulary without taking word frequency into account; in real life you might want to use Keras vectorizer, and limit the number of words.")

vocab = set(df['Word'].apply(lambda x: x.lower()))
id2word = { i+1 : v for i,v in enumerate(vocab) }
id2word[0] = '<UNK>'
vocab.add('<UNK>')
word2id = { v : k for k,v in id2word.items() }

"""
We need to create a dataset of sentences for training. Let's loop through the original dataset and separate all individual sentences into `X` (lists of words) and `Y` (list of tokens):
"""
logger.info("We need to create a dataset of sentences for training. Let's loop through the original dataset and separate all individual sentences into `X` (lists of words) and `Y` (list of tokens):")

X,Y = [],[]
s,t = [],[]
for i,row in df[['Sentence #','Word','Tag']].iterrows():
    if pd.isna(row['Sentence #']):
        s.append(row['Word'])
        t.append(row['Tag'])
    else:
        if len(s)>0:
            X.append(s)
            Y.append(t)
        s,t = [row['Word']],[row['Tag']]
X.append(s)
Y.append(t)

"""
We will now vectorize all words and tokens:
"""
logger.info("We will now vectorize all words and tokens:")

def vectorize(seq):
    return [word2id[x.lower()] for x in seq]

def tagify(seq):
    return [tag2id[x] for x in seq]

Xv = list(map(vectorize,X))
Yv = list(map(tagify,Y))

Xv[0], Yv[0]

"""
For simplicity, we will pad all sentences with 0 tokens to the maximum length. In real life, we might want to use more clever strategy, and pad sequences only within one minibatch.
"""
logger.info("For simplicity, we will pad all sentences with 0 tokens to the maximum length. In real life, we might want to use more clever strategy, and pad sequences only within one minibatch.")

X_data = keras.preprocessing.sequence.pad_sequences(Xv,padding='post')
Y_data = keras.preprocessing.sequence.pad_sequences(Yv,padding='post')

"""
## Defining Token Classification Network

We will use two-layer bidirectional LSTM network for token classification. In order to apply dense classifier to each of the output of the last LSTM layer, we will use `TimeDistributed` construction, which replicates the same dense layer to each of the outputs of LSTM at each step:
"""
logger.info("## Defining Token Classification Network")

maxlen = X_data.shape[1]
vocab_size = len(vocab)
num_tags = len(tags)
model = keras.models.Sequential([
    keras.layers.Embedding(vocab_size, 300, input_length=maxlen),
    keras.layers.Bidirectional(keras.layers.LSTM(units=100, activation='tanh', return_sequences=True)),
    keras.layers.Bidirectional(keras.layers.LSTM(units=100, activation='tanh', return_sequences=True)),
    keras.layers.TimeDistributed(keras.layers.Dense(num_tags, activation='softmax'))
])
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['acc'])
model.summary()

"""
Note here that we are explicity specifying `maxlen` for our dataset - in case we want the network to be able to handle variable length sequences, we need to be a bit more clever when defining the network.

Let's now train the model. For speed, we will only train for one epoch, but you may try training for longer time. Also, you may want to separate some part of the dataset as training dataset, to observe validation accuracy.
"""
logger.info("Note here that we are explicity specifying `maxlen` for our dataset - in case we want the network to be able to handle variable length sequences, we need to be a bit more clever when defining the network.")

model.fit(X_data,Y_data)

"""
## Testing the Result

Let's now see how our entity recognition model works on a sample sentence:
"""
logger.info("## Testing the Result")

sent = 'John Smith went to Paris to attend a conference in cancer development institute'
words = sent.lower().split()
v = keras.preprocessing.sequence.pad_sequences([[word2id[x] for x in words]],padding='post',maxlen=maxlen)
res = model(v)[0]

r = np.argmax(res.numpy(),axis=1)
for i,w in zip(r,words):
    logger.debug(f"{w} -> {id2tag[i]}")

"""
## Takeaway

Even simple LSTM model shows reasonable results at NER. However, to get much better results, you may want to use large pre-trained language models such as BERT. Training BERT for NER using Huggingface Transformers library is described [here](https://huggingface.co/course/chapter7/2?fw=pt).
"""
logger.info("## Takeaway")

logger.info("\n\n[DONE]", bright=True)