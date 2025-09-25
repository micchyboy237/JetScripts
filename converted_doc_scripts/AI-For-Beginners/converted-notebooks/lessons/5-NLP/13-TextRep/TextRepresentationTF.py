from jet.logger import logger
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow import keras
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
# Text classification task

In this module, we will start with a simple text classification task based on the **[AG_NEWS](http://www.di.unipi.it/~gulli/AG_corpus_of_news_articles.html)** dataset: we'll classify news headlines into one of 4 categories: World, Sports, Business and Sci/Tech. 

## The Dataset

To load the dataset, we will use the **[TensorFlow Datasets](https://www.tensorflow.org/datasets)** API.
"""
logger.info("# Text classification task")


physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices)>0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

dataset = tfds.load('ag_news_subset')

"""
We can now access the training and test portions of the dataset by using `dataset['train']` and `dataset['test']` respectively:
"""
logger.info("We can now access the training and test portions of the dataset by using `dataset['train']` and `dataset['test']` respectively:")

ds_train = dataset['train']
ds_test = dataset['test']

logger.debug(f"Length of train dataset = {len(ds_train)}")
logger.debug(f"Length of test dataset = {len(ds_test)}")

"""
Let's print out the first 10 new headlines from our dataset:
"""
logger.info("Let's print out the first 10 new headlines from our dataset:")

classes = ['World', 'Sports', 'Business', 'Sci/Tech']

for i,x in zip(range(5),ds_train):
    logger.debug(f"{x['label']} ({classes[x['label']]}) -> {x['title']} {x['description']}")

"""
## Text vectorization

Now we need to convert text into **numbers** that can be represented as tensors. If we want word-level representation, we need to do two things:

* Use a **tokenizer** to split text into **tokens**.
* Build a **vocabulary** of those tokens.

### Limiting vocabulary size

In the AG News dataset example, the vocabulary size is rather big, more than 100k words. Generally speaking, we don't need words that are rarely present in the text &mdash; only a few sentences will have them, and the model will not learn from them. Thus, it makes sense to limit the vocabulary size to a smaller number by passing an argument to the vectorizer constructor:

Both of those steps can be handled using the **TextVectorization** layer. Let's instantiate the vectorizer object, and then call the `adapt` method to go through all text and build a vocabulary:
"""
logger.info("## Text vectorization")

vocab_size = 50000
vectorizer = keras.layers.experimental.preprocessing.TextVectorization(max_tokens=vocab_size)
vectorizer.adapt(ds_train.take(500).map(lambda x: x['title']+' '+x['description']))

"""
> **Note** that we are using only subset of the whole dataset to build a vocabulary. We do it to speed up the execution time and not keep you waiting. However, we are taking the risk that some of the words from the whole dateset would not be included into the vocabulary, and will be ignored during training. Thus, using the whole vocabulary size and running through all dataset during `adapt` should increase the final accuracy, but not significantly.

Now we can access the actual vocabulary:
"""
logger.info("Now we can access the actual vocabulary:")

vocab = vectorizer.get_vocabulary()
vocab_size = len(vocab)
logger.debug(vocab[:10])
logger.debug(f"Length of vocabulary: {vocab_size}")

"""
Using the vectorizer, we can easily encode any text into a set of numbers:
"""
logger.info("Using the vectorizer, we can easily encode any text into a set of numbers:")

vectorizer('I love to play with my words')

"""
## Bag-of-words text representation

Because words represent meaning, sometimes we can figure out the meaning of a piece of text by just looking at the individual words, regardless of their order in the sentence. For example, when classifying news, words like *weather* and *snow* are likely to indicate *weather forecast*, while words like *stocks* and *dollar* would count towards *financial news*.

**Bag-of-words** (BoW) vector representation is the most simple to understand traditional vector representation. Each word is linked to a vector index, and a vector element contains the number of occurrences of each word in a given document.

![Image showing how a bag of words vector representation is represented in memory.](images/bag-of-words-example.png) 

> **Note**: You can also think of BoW as a sum of all one-hot-encoded vectors for individual words in the text.

Below is an example of how to generate a bag-of-words representation using the Scikit Learn python library:
"""
logger.info("## Bag-of-words text representation")

sc_vectorizer = CountVectorizer()
corpus = [
        'I like hot dogs.',
        'The dog ran fast.',
        'Its hot outside.',
    ]
sc_vectorizer.fit_transform(corpus)
sc_vectorizer.transform(['My dog likes hot dogs on a hot day.']).toarray()

"""
We can also use the Keras vectorizer that we defined above, converting each word number into a one-hot encoding and adding all those vectors up:
"""
logger.info("We can also use the Keras vectorizer that we defined above, converting each word number into a one-hot encoding and adding all those vectors up:")

def to_bow(text):
    return tf.reduce_sum(tf.one_hot(vectorizer(text),vocab_size),axis=0)

to_bow('My dog likes hot dogs on a hot day.').numpy()

"""
> **Note**: You may be surprised that the result differs from the previous example. The reason is that in the Keras example the length of the vector corresponds to the vocabulary size, which was built from the whole AG News dataset, while in the Scikit Learn example we built the vocabulary from the sample text on the fly.

## Training the BoW classifier

Now that we have learned how to build the bag-of-words representation of our text, let's train a classifier that uses it. First, we need to convert our dataset to a bag-of-words representation. This can be achieved by using `map` function in the following way:
"""
logger.info("## Training the BoW classifier")

batch_size = 128

ds_train_bow = ds_train.map(lambda x: (to_bow(x['title']+x['description']),x['label'])).batch(batch_size)
ds_test_bow = ds_test.map(lambda x: (to_bow(x['title']+x['description']),x['label'])).batch(batch_size)

"""
Now let's define a simple classifier neural network that contains one linear layer. The input size is `vocab_size`, and the output size corresponds to the number of classes (4). Because we're solving a classification task, the final activation function is **softmax**:
"""
logger.info("Now let's define a simple classifier neural network that contains one linear layer. The input size is `vocab_size`, and the output size corresponds to the number of classes (4). Because we're solving a classification task, the final activation function is **softmax**:")

model = keras.models.Sequential([
    keras.layers.Dense(4,activation='softmax',input_shape=(vocab_size,))
])
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(ds_train_bow,validation_data=ds_test_bow)

"""
Since we have 4 classes, an accuracy of above 80% is a good result.

## Training a classifier as one network

Because the vectorizer is also a Keras layer, we can define a network that includes it, and train it end-to-end. This way we don't need to vectorize the dataset using `map`, we can just pass the original dataset to the input of the network.

> **Note**: We would still have to apply maps to our dataset to convert fields from dictionaries (such as `title`, `description` and `label`) to tuples. However, when loading data from disk, we can build a dataset with the required structure in the first place.
"""
logger.info("## Training a classifier as one network")

def extract_text(x):
    return x['title']+' '+x['description']

def tupelize(x):
    return (extract_text(x),x['label'])

inp = keras.Input(shape=(1,),dtype=tf.string)
x = vectorizer(inp)
x = tf.reduce_sum(tf.one_hot(x,vocab_size),axis=1)
out = keras.layers.Dense(4,activation='softmax')(x)
model = keras.models.Model(inp,out)
model.summary()

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(ds_train.map(tupelize).batch(batch_size),validation_data=ds_test.map(tupelize).batch(batch_size))

"""
## Bigrams, trigrams and n-grams

One limitation of the bag-of-words approach is that some words are part of multi-word expressions, for example, the word 'hot dog' has a completely different meaning from the words 'hot' and 'dog' in other contexts. If we represent the words 'hot' and 'dog' always using the same vectors, it can confuse our model.

To address this, **n-gram representations** are often used in methods of document classification, where the frequency of each word, bi-word or tri-word is a useful feature for training classifiers. In bigram representations, for example, we will add all word pairs to the vocabulary, in addition to original words.

Below is an example of how to generate a bigram bag of word representation using Scikit Learn:
"""
logger.info("## Bigrams, trigrams and n-grams")

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
The main drawback of the n-gram approach is that the vocabulary size starts to grow extremely fast. In practice, we need to combine the n-gram representation with a dimensionality reduction technique, such as *embeddings*, which we will discuss in the next unit.

To use an n-gram representation in our **AG News** dataset, we need to pass the `ngrams` parameter to our `TextVectorization` constructor. The length of a bigram vocaculary is **significantly larger**, in our case it is more than 1.3 million tokens! Thus it makes sense to limit bigram tokens as well by some reasonable number.

We could use the same code as above to train the classifier, however, it would be very memory-inefficient. In the next unit, we will train the bigram classifier using embeddings. In the meantime, you can experiment with bigram classifier training in this notebook and see if you can get higher accuracy.

## Automatically calculating BoW Vectors

In the example above we calculated BoW vectors by hand by summing the one-hot encodings of individual words. However, the latest version of TensorFlow allows us to calculate BoW vectors automatically by passing the `output_mode='count` parameter to the vectorizer constructor. This makes defining and training our model significanly easier:
"""
logger.info("## Automatically calculating BoW Vectors")

model = keras.models.Sequential([
    keras.layers.experimental.preprocessing.TextVectorization(max_tokens=vocab_size,output_mode='count'),
    keras.layers.Dense(4,input_shape=(vocab_size,), activation='softmax')
])
logger.debug("Training vectorizer")
model.layers[0].adapt(ds_train.take(500).map(extract_text))
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(ds_train.map(tupelize).batch(batch_size),validation_data=ds_test.map(tupelize).batch(batch_size))

"""
## Term frequency - inverse document frequency (TF-IDF)

In BoW representation, word occurrences are weighted using the same technique regardless of the word itself. However, it's clear that frequent words such as *a* and *in* are much less important for classification than specialized terms. In most NLP tasks some words are more relevant than others.

**TF-IDF** stands for **term frequency - inverse document frequency**. It's a variation of bag-of-words, where instead of a binary 0/1 value indicating the appearance of a word in a document, a floating-point value is used, which is related to the frequency of the word occurrence in the corpus.

More formally, the weight $w_{ij}$ of a word $i$ in the document $j$ is defined as:
$$
w_{ij} = tf_{ij}\times\log({N\over df_i})
$$
where
* $tf_{ij}$ is the number of occurrences of $i$ in $j$, i.e. the BoW value we have seen before
* $N$ is the number of documents in the collection
* $df_i$ is the number of documents containing the word $i$ in the whole collection

The TF-IDF value $w_{ij}$ increases proportionally to the number of times a word appears in a document and is offset by the number of documents in the corpus that contains the word, which helps to adjust for the fact that some words appear more frequently than others. For example, if the word appears in *every* document in the collection, $df_i=N$, and $w_{ij}=0$, and those terms would be completely disregarded.

You can easily create TF-IDF vectorization of text using Scikit Learn:
"""
logger.info("## Term frequency - inverse document frequency (TF-IDF)")

vectorizer = TfidfVectorizer(ngram_range=(1,2))
vectorizer.fit_transform(corpus)
vectorizer.transform(['My dog likes hot dogs on a hot day.']).toarray()

"""
In Keras, the `TextVectorization` layer can automatically compute TF-IDF frequencies by passing the `output_mode='tf-idf'` parameter. Let's repeat the code we used above to see if using TF-IDF increases accuracy:
"""
logger.info("In Keras, the `TextVectorization` layer can automatically compute TF-IDF frequencies by passing the `output_mode='tf-idf'` parameter. Let's repeat the code we used above to see if using TF-IDF increases accuracy:")

model = keras.models.Sequential([
    keras.layers.experimental.preprocessing.TextVectorization(max_tokens=vocab_size,output_mode='tf-idf'),
    keras.layers.Dense(4,input_shape=(vocab_size,), activation='softmax')
])
logger.debug("Training vectorizer")
model.layers[0].adapt(ds_train.take(500).map(extract_text))
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(ds_train.map(tupelize).batch(batch_size),validation_data=ds_test.map(tupelize).batch(batch_size))

"""
## Conclusion 

Even though TF-IDF representations provide frequency weights to different words, they are unable to represent meaning or order. As the famous linguist J. R. Firth said in 1935, "The complete meaning of a word is always contextual, and no study of meaning apart from context can be taken seriously." We will learn how to capture contextual information from text using language modeling later in the course.
"""
logger.info("## Conclusion")

logger.info("\n\n[DONE]", bright=True)