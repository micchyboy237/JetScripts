from jet.logger import logger
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os
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
## The Effect of Dropout

Let's see for ourselves how dropout actually affects training. We will use MNIST dataset and a simple convolutional network to do that:
"""
logger.info("## The Effect of Dropout")


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

"""
We will define `train` function that will take care of all training process, including:
* Defining the neural network architecture with a given dropout rate `d`
* Specifying suitable training parameters (optimizer and loss function)
* Doing the training and collecting the history

We will then run this function for a bunch of different dropout values:
"""
logger.info("We will define `train` function that will take care of all training process, including:")

def train(d):
    logger.debug(f"Training with dropout = {d}")
    model = keras.Sequential([
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28,28,1)),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dropout(d),
        keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['acc'])
    hist = model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=5,batch_size=64)
    return hist

res = { d : train(d) for d in [0,0.2,0.5,0.8] }

"""
Now, let's plot validation accuracy graphs for different dropout values to see how fast the training goes:
"""
logger.info("Now, let's plot validation accuracy graphs for different dropout values to see how fast the training goes:")

for d,h in res.items():
    plt.plot(h.history['val_acc'],label=str(d))
plt.legend()

"""
From this graph, you would probably be able to see the following:
* Dropout values in the 0.2-0.5 range, you will see the fastest training the best overall results
* Without dropout ($d=0$), you are likely to see less stable and slower training process
* High dropout (0.8) makes things worse
"""
logger.info("From this graph, you would probably be able to see the following:")

logger.info("\n\n[DONE]", bright=True)