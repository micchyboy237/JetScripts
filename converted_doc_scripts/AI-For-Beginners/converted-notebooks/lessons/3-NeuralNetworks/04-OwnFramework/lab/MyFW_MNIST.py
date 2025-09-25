from jet.logger import logger
from sklearn.model_selection import train_test_split
import os
import pickle
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
# MNIST Digit Classification with our own Framework

Lab Assignment from [AI for Beginners Curriculum](https://github.com/microsoft/ai-for-beginners).

### Reading the Dataset

This code download the dataset from the repository on the internet. You can also manually copy the dataset from `/data` directory of AI Curriculum repo.
"""
logger.info("# MNIST Digit Classification with our own Framework")

# !rm *.pkl
# !wget https://raw.githubusercontent.com/microsoft/AI-For-Beginners/main/data/mnist.pkl.gz
# !gzip -d mnist.pkl.gz

with open('mnist.pkl','rb') as f:
    MNIST = pickle.load(f)

labels = MNIST['Train']['Labels']
data = MNIST['Train']['Features']

"""
Let's see what is the shape of data that we have:
"""
logger.info("Let's see what is the shape of data that we have:")

data.shape

"""
### Splitting the Data

We will use Scikit Learn to split the data between training and test dataset:
"""
logger.info("### Splitting the Data")


features_train, features_test, labels_train, labels_test = train_test_split(data,labels,test_size=0.2)

logger.debug(f"Train samples: {len(features_train)}, test samples: {len(features_test)}")

"""
### Instructions

1. Take the framework code from the lesson and paste it into this notebook, or (even better) into a separate Python module
1. Define and train one-layered perceptron, observing training and validation accuracy during training
1. Try to understand if overfitting took place, and adjust layer parameters to improve accuracy
1. Repeat previous steps for 2- and 3-layered perceptrons. Try to experiment with different activation functions between layers.
1. Try to answer the following questions:
    - Does the inter-layer activation function affect network performance?
    - Do we need 2- or 3-layered network for this task?
    - Did you experience any problems training the network? Especially as the number of layers increased.
    - How do weights of the network behave during training? You may plot max abs value of weights vs. epoch to understand the relation.
"""
logger.info("### Instructions")


logger.info("\n\n[DONE]", bright=True)