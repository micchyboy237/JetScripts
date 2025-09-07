from imblearn.under_sampling import RandomUnderSampler
from jet.logger import CustomLogger
from pymongo import MongoClient
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from transformers import AutoConfig, AutoTokenizer, TFAutoModelForSequenceClassification
import numpy as np
import os
import pandas as pd
import shutil
import tensorflow as tf
import tensorflow_hub as hub


OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "generated", os.path.splitext(os.path.basename(__file__))[0])
shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
LOG_DIR = f"{OUTPUT_DIR}/logs"

log_file = os.path.join(LOG_DIR, "main.log")
logger = CustomLogger(log_file, overwrite=True)
logger.orange(f"Logs: {log_file}")

"""
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mongodb-developer/GenAI-Showcase/blob/main/notebooks/ml/tensorflow_mongodbcharts_horoscopes.ipynb)

[![View Article](https://img.shields.io/badge/View%20Article-blue)](https://www.mongodb.com/developer/products/mongodb/tensorflow-mongodb-charts/)

# Sentiment Analysis on my scraped horoscopes

Our first step is to do sentiment analysis on our .csv file of scraped horoscopes. Luckily for us, at this point in the tutorial we don't need to build a model (yet!), we can use a pre-trained model to figure out whether or not our horoscopes from the past six months are positive or negative.

I am using this tutorial here from [Medium](https://medium.com/@sharma.tanish096/sentiment-analysis-using-pre-trained-models-and-transformer-28e9b9486641), please feel free to take a look at it to better understand the code used below.
"""
logger.info("# Sentiment Analysis on my scraped horoscopes")


"""
Once everything is imported in, let's choose which pre-trained model we want to use. Since this is a TensorFlow tutorial, we can go ahead and use the ["distilbert-base-uncased-finetuned-sst-2-english"](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) model since it's compatible with TensorFlow, but there are a ton of options out there to choose from if you would like to switch it up.
"""
logger.info("Once everything is imported in, let's choose which pre-trained model we want to use. Since this is a TensorFlow tutorial, we can go ahead and use the ["distilbert-base-uncased-finetuned-sst-2-english"](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) model since it's compatible with TensorFlow, but there are a ton of options out there to choose from if you would like to switch it up.")

distilbert = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(distilbert)
config = AutoConfig.from_pretrained(distilbert)
model = TFAutoModelForSequenceClassification.from_pretrained(distilbert)


def sentiment_finder(horoscope):
    input = tokenizer(
        horoscope, padding=True, truncation=True, max_length=512, return_tensors="tf"
    )
    output = model(input)
    scores = output.logits[0].numpy()
    scores = softmax(scores)
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    return config.id2label[ranking[0]]


horoscope = "Things might get a bit confusing for you today, Capricorn. Don't feel like you need to make sense of it all. In fact, this task may be impossible. Just be yourself. Let your creative nature shine through. Other people are quite malleable, and you should feel free to take the lead in just about any situation. Make sure, however, that you consider other people's needs."
sentiment = sentiment_finder(horoscope)
logger.debug(f"Horoscope is {sentiment}")

"""
This is great! As we can see, that Capricorn horoscope is in fact negative, and we were able to use our pre-trained model to classify it. But, now we need to make some changes because we don't want to put in everything manually, we want to use this pre-trained model and put in our .csv file of all our horoscopes and figure out the sentiment analysis of everything in our file, while also incorporating in a new "sentiment" column that will include 1's if the horoscope is positive and 0 if the horoscope is negative.
"""
logger.info("This is great! As we can see, that Capricorn horoscope is in fact negative, and we were able to use our pre-trained model to classify it. But, now we need to make some changes because we don't want to put in everything manually, we want to use this pre-trained model and put in our .csv file of all our horoscopes and figure out the sentiment analysis of everything in our file, while also incorporating in a new "sentiment" column that will include 1's if the horoscope is positive and 0 if the horoscope is negative.")

def apply_sentiment(horoscope):
    sentiment = sentiment_finder(horoscope)
    return 1 if sentiment == "POSITIVE" else 0

"""
Now load our "anaiya-six-months-horoscopes.csv" file
"""
logger.info("Now load our "anaiya-six-months-horoscopes.csv" file")

# !pip install pandas

df = pd.read_csv("anaiya-six-months-horoscopes.csv")

df["sentiment"] = df["horoscope"].apply(apply_sentiment)

df.to_csv("anaiya-six-months-horoscopes-sentiment.csv")

logger.debug("saved to new file called anaiya-six-months-horoscopes-sentiment.csv")

df.head()

"""
## Save our new `.csv` file into MongoDB Atlas so we can visualize our data in MongoDB Charts

This part is done using MongoDB Compass and MongoDB Charts.

# TRAIN AND TEST MODEL WITH TENSORFLOW

Now that we have our dataset ready with our sentiment analysis done using our pre-trained model, we can go ahead and set up a way to train and test our data so that if we wanted to incorporate new horoscopes, we can see if they will be negative or positive.

In order to help me learn how to do this, I watched this video from freeCodeCamp.org: https://www.youtube.com/watch?v=VtRLrQ3Ev-U, and I used the skeleton code from this TensorFlow docs: https://www.tensorflow.org/tutorials/structured_data/preprocessing_layers

Feel free to watch it to get a better understanding of the code used below.
"""
logger.info("## Save our new `.csv` file into MongoDB Atlas so we can visualize our data in MongoDB Charts")


df = pd.read_csv("anaiya-six-months-horoscopes-sentiment.csv")
df = df[["horoscope", "sentiment"]]

"""
We want to split up our dataset into three sets. We need a training set, a validation set, and a test set.

# BALANCE DATASET
We need to balance our dataset since we need to make sure our model is trained on the same exact amount of negative and positive horoscopes, otherwise things will be swayed in one direction or the other. Check out this article for help on how to balance your dataset: https://medium.com/@daniele.santiago/balancing-imbalanced-data-undersampling-and-oversampling-techniques-in-python-7c5378282290 and https://semaphoreci.com/blog/imbalanced-data-machine-learning-python
"""
logger.info("# BALANCE DATASET")

df = df.sample(frac=1, random_state=42)

X = df["horoscope"]
y = df["sentiment"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, shuffle=True, test_size=0.15, random_state=42
)

rus = RandomUnderSampler(random_state=42, sampling_strategy="majority")

X_resampled, y_resampled = rus.fit_resample(X_train.to_frame(), y_train)

balanced_trained = pd.DataFrame(
    {"horoscope": X_resampled["horoscope"], "sentiment": y_resampled}
)

sentiment_amount_training = balanced_trained["sentiment"].value_counts()
logger.debug(sentiment_amount_training)

"""
# SPLIT UP OUR DATASET
"""
logger.info("# SPLIT UP OUR DATASET")

train, val = train_test_split(
    balanced_trained,
    test_size=0.2,
    stratify=balanced_trained["sentiment"],
    random_state=42,
)

logger.debug("Training set:", len(train))
logger.debug("Validation set:", len(val))
logger.debug("Test set:", len(X_test))

test = pd.DataFrame({"horoscope": X_test, "sentiment": y_test})

"""
# CONVERT TO TENSORFLOW DATASET

Now, let's convert our dataframes to TensorFlow datasets. Use this code from the documentation: https://www.tensorflow.org/tutorials/structured_data/preprocessing_layers

it converts each train, validation, and test dataset into a tensorflow dataset and will shuffle again and batch the data for you.
"""
logger.info("# CONVERT TO TENSORFLOW DATASET")

def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    df = dataframe.copy()
    labels = df.pop("target")
    df = {key: value.values[:, tf.newaxis] for key, value in dataframe.items()}
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds

def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    df = dataframe.copy()
    labels = df.pop("sentiment")
    df = df["horoscope"]
    ds = tf.data.Dataset.from_tensor_slices((df, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_data = df_to_dataset(train)
val_data = df_to_dataset(val)
test_data = df_to_dataset(test)

list(train_data)[0]

"""
Now we want to embed and build our model

# EMBEDDING LAYER
"""
logger.info("# EMBEDDING LAYER")

embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(embedding, dtype=tf.string, trainable=True)

"""
# MODEL

Now, we need to build out our neural network model.
We want various layers here built out with the Sequential model since it's a way of stacking the layers one by one, and is the easiest model to understand and visualize. We are also going to be using Dropout layers since it's a good way to prevent overfitting, which can lead your model astray. We are going to be using a dropout of 0.4, 0.3 and 0.2, so 40%, 30% and 20% of our neural networks neurons will be randomly dropped out, or set to zero, so that our model can work better.
"""
logger.info("# MODEL")

model = tf.keras.Sequential()  # since layer by layer so sequential. most basic form
model.add(hub_layer)
model.add(tf.keras.layers.Dense(128, activation="relu"))  # first neural network layer
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(64, activation="relu"))  # second layer
model.add(tf.keras.layers.Dropout(0.3))  # another dropout layer
model.add(tf.keras.layers.Dense(32, activation="relu"))  # third layer
model.add(tf.keras.layers.Dropout(0.2))  # another dropout layer
model.add(tf.keras.layers.Dense(1, activation="sigmoid"))  # output layer

"""
Now we want to compile our model
"""
logger.info("Now we want to compile our model")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=["accuracy"],
)

"""
Let's now train our model on our training data
"""
logger.info("Let's now train our model on our training data")

history = model.fit(train_data, epochs=5, validation_data=val_data)

"""
Once again we see that the loss may be plateauing in some places, but is overall decreasing. We can see that our val_accuracy is increasing so this means that our model is being trained nicely on our dataset.

Now, evaluate our model on our test dataset
"""
logger.info("Once again we see that the loss may be plateauing in some places, but is overall decreasing. We can see that our val_accuracy is increasing so this means that our model is being trained nicely on our dataset.")

loss, accuracy = model.evaluate(test_data)
logger.debug(f"Loss: {loss}")
logger.debug(f"Accuracy: {accuracy}")

"""
# NEW HOROSCOPE PREDICTION

https://www.tensorflow.org/api_docs/python/tf/squeeze

tf.squeeze is how you can get the probability from our prediction
"""
logger.info("# NEW HOROSCOPE PREDICTION")

def predict_sentiment(horoscope):
    encoded_input = tf.constant([horoscope])

    prediction = model.predict(encoded_input)

    probability = tf.squeeze(prediction).numpy()
    logger.debug(f"model probability: {probability}")

    sentiment = 1 if probability > 0.5 else 0

    return sentiment


positive_horoscope = "You're incredibly productive, with good business sense, Libra."
negative_horoscope = "This isn't the most cheerful time, Leo, because important issues are rearing their heads again and forcing you to address them."
pos_sentiment = predict_sentiment(positive_horoscope)
neg_sentiment = predict_sentiment(negative_horoscope)

logger.debug(f"This should be positive: {pos_sentiment}")
logger.debug(f"This should be negative: {neg_sentiment}")

"""
## Let's see how our week will be going forward
"""
logger.info("## Let's see how our week will be going forward")

file = "new-week-horoscopes2.csv"
df = pd.read_csv(file)

df["sentiment"] = df["horoscope"].apply(predict_sentiment)

for index, row in df.iterrows():
    zodiac = row["zodiac"]
    horoscope = row["horoscope"]
    sentiment = row["sentiment"]
    logger.debug(f"{zodiac} horoscope is {sentiment}")

"""
## lets save these back into MongoDB Atlas so we can visualize them in Charts
"""
logger.info("## lets save these back into MongoDB Atlas so we can visualize them in Charts")

pip install pymongo

# import getpass


# connection_string = getpass.getpass(
    prompt="Enter connection string WITH USER + PASS here"
)
client = MongoClient(
    connection_string, appname="devrel.showcase.tensorflow_mongodbcharts"
)

database = client["horoscopes"]
collection = database["new_week_horoscope"]

for index, row in df.iterrows():
    zodiac = row["zodiac"]
    horoscope = row["horoscope"]
    sentiment = row["sentiment"]

    dict = {"zodiac": zodiac, "horoscope": horoscope, "sentiment": sentiment}

    collection.insert_one(dict)


logger.debug("saved in! go check")

logger.info("\n\n[DONE]", bright=True)