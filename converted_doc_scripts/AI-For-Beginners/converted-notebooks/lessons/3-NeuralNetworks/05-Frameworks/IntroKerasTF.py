from jet.logger import logger
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import shutil
import tensorflow as tf


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
## Introduction to Tensorflow and Keras

> This notebook is a part of [AI for Beginners Curricula](http://github.com/microsoft/ai-for-beginners). Visit the repository for complete set of learning materials.

### Neural Frameworks

We have learnt that to train neural networks you need:
* Quickly multiply matrices (tensors)
* Compute gradients to perform gradient descent optimization

What neural network frameworks allow you to do:
* Operate with tensors on whatever compute is available, CPU or GPU, or even TPU
* Automatically compute gradients (they are explicitly programmed for all built-in tensor functions)

Optionally:
* Neural Network constructor / higher level API (describe network as a sequence of layers)
* Simple training functions (`fit`, as in Scikit Learn)
* A number of optimization algorithms in addition to gradient descent
* Data handling abstractions (that will ideally work on GPU, too)

### Most Popular Frameworks

* Tensorflow 1.x - first widely available framework (Google). Allowed to define static computation graph, push it to GPU, and explicitly evaluate it
* PyTorch - a framework from Facebook that is growing in popularity
* Keras - higher level API on top of Tensorflow/PyTorch to unify and simplify using neural networks (Francois Chollet)
* Tensorflow 2.x + Keras - new version of Tensorflow with integrated Keras functionality, which supports **dynamic computation graph**, allowing to perform tensor operations very similar to numpy (and PyTorch)

We will consider Tensorflow 2.x and Keras. Make sure you have version 2.x.x of Tensorflow installed:
```
pip install tensorflow
```
or
```
conda install tensorflow
```
"""
logger.info("## Introduction to Tensorflow and Keras")

logger.debug(tf.__version__)

"""
## Basic Concepts: Tensor

**Tensor** is a multi-dimensional array. It is very convenient to use tensors to represent different types of data:
* 400x400 - black-and-white picture
* 400x400x3 - color picture 
* 16x400x400x3 - minibatch of 16 color pictures
* 25x400x400x3 - one second of 25-fps video
* 8x25x400x400x3 - minibatch of 8 1-second videos

### Simple Tensors

You can easily create simple tensors from lists of np-arrays, or generate random ones:
"""
logger.info("## Basic Concepts: Tensor")

a = tf.constant([[1,2],[3,4]])
logger.debug(a)
a = tf.random.normal(shape=(10,3))
logger.debug(a)

"""
You can use arithmetic operations on tensors, which are performed element-wise, as in numpy. Tensors are automatically expanded to required dimension, if needed. To extract numpy-array from tensor, use `.numpy()`:
"""
logger.info("You can use arithmetic operations on tensors, which are performed element-wise, as in numpy. Tensors are automatically expanded to required dimension, if needed. To extract numpy-array from tensor, use `.numpy()`:")

logger.debug(a-a[0])
logger.debug(tf.exp(a)[0].numpy())

"""
## Variables

Variables are useful to represent tensor values that can be modified using `assign` and `assign_add`. They are often used to represent neural network weights.

As an example, here is a silly way to get a sum of all rows of tensor `a`:
"""
logger.info("## Variables")

s = tf.Variable(tf.zeros_like(a[0]))
for i in a:
  s.assign_add(i)

logger.debug(s)

"""
Much better way to do it:
"""
logger.info("Much better way to do it:")

tf.reduce_sum(a,axis=0)

"""
## Computing Gradients

For back propagation, you need to compute gradients. This is done using `tf.GradientTape()` idiom:
 * Add `with tf.GradientTape` block around our computations
 * Mark those tensors with respect to which we need to compute gradients by calling `tape.watch` (all variables are watched automatically)
 * Compute whatever we need (build computational graph)
 * Obtain gradients using `tape.gradient`
"""
logger.info("## Computing Gradients")

a = tf.random.normal(shape=(2, 2))
b = tf.random.normal(shape=(2, 2))

with tf.GradientTape() as tape:
  tape.watch(a)  # Start recording the history of operations applied to `a`
  c = tf.sqrt(tf.square(a) + tf.square(b))  # Do some math using `a`
  dc_da = tape.gradient(c, a)
  logger.debug(dc_da)

"""
## Example 1: Linear Regression

Now we know enough to solve the classical problem of **Linear regression**. Let's generate small synthetic dataset:
"""
logger.info("## Example 1: Linear Regression")


np.random.seed(13) # pick the seed for reproducability - change it to explore the effects of random variations

train_x = np.linspace(0, 3, 120)
train_labels = 2 * train_x + 0.9 + np.random.randn(*train_x.shape) * 0.5

plt.scatter(train_x,train_labels)

"""
Linear regression is defined by a straight line $f_{W,b}(x) = Wx+b$, where $W, b$ are model parameters that we need to find. An error on our dataset $\{x_i,y_u\}_{i=1}^N$ (also called **loss function**) can be defined as mean square error:
$$
\mathcal{L}(W,b) = {1\over N}\sum_{i=1}^N (f_{W,b}(x_i)-y_i)^2
$$

Let's define our model and loss function:
"""
logger.info("Linear regression is defined by a straight line $f_{W,b}(x) = Wx+b$, where $W, b$ are model parameters that we need to find. An error on our dataset $\{x_i,y_u\}_{i=1}^N$ (also called **loss function**) can be defined as mean square error:")

input_dim = 1
output_dim = 1
learning_rate = 0.1

w = tf.Variable([[100.0]])
b = tf.Variable(tf.zeros(shape=(output_dim,)))

def f(x):
  return tf.matmul(x,w) + b

def compute_loss(labels, predictions):
  return tf.reduce_mean(tf.square(labels - predictions))

"""
We will train the model on a series of minibatches. We will use gradient descent, adjusting model parameters using the following formulae:
$$
\begin{array}{l}
W^{(n+1)}=W^{(n)}-\eta\frac{\partial\mathcal{L}}{\partial W} \\
b^{(n+1)}=b^{(n)}-\eta\frac{\partial\mathcal{L}}{\partial b} \\
\end{array}
$$
"""
logger.info("We will train the model on a series of minibatches. We will use gradient descent, adjusting model parameters using the following formulae:")

def train_on_batch(x, y):
  with tf.GradientTape() as tape:
    predictions = f(x)
    loss = compute_loss(y, predictions)
    dloss_dw, dloss_db = tape.gradient(loss, [w, b])
  w.assign_sub(learning_rate * dloss_dw)
  b.assign_sub(learning_rate * dloss_db)
  return loss

"""
Let's do the training. We will do several passes through the dataset (so-called **epochs**), divide it into minibatches and call the function defined above:
"""
logger.info("Let's do the training. We will do several passes through the dataset (so-called **epochs**), divide it into minibatches and call the function defined above:")

indices = np.random.permutation(len(train_x))
features = tf.constant(train_x[indices],dtype=tf.float32)
labels = tf.constant(train_labels[indices],dtype=tf.float32)

batch_size = 4
for epoch in range(10):
  for i in range(0,len(features),batch_size):
    loss = train_on_batch(tf.reshape(features[i:i+batch_size],(-1,1)),tf.reshape(labels[i:i+batch_size],(-1,1)))
  logger.debug('Epoch %d: last batch loss = %.4f' % (epoch, float(loss)))

"""
We now have obtained optimized parameters $W$ and $b$. Note that their values are similar to the original values used when generating the dataset ($W=2, b=1$)
"""
logger.info("We now have obtained optimized parameters $W$ and $b$. Note that their values are similar to the original values used when generating the dataset ($W=2, b=1$)")

w,b

plt.scatter(train_x,train_labels)
x = np.array([min(train_x),max(train_x)])
y = w.numpy()[0,0]*x+b.numpy()[0]
plt.plot(x,y,color='red')

"""
## Computational Graph and GPU Computations

Whenever we compute tensor expression, Tensorflow builds a computational graph that can be computed on the available computing device, e.g. CPU or GPU. Since we were using arbitrary Python function in our code, they cannot be included as part of computational graph, and thus when running our code on GPU we would need to pass the data between CPU and GPU back and forth, and compute custom function on CPU.

Tensorflow allows us to mark our Python function using `@tf.function` decorator, which will make this function a part of the same computational graph. This decorator can be applied to functions that use standard Tensorflow tensor operations.
"""
logger.info("## Computational Graph and GPU Computations")

@tf.function
def train_on_batch(x, y):
  with tf.GradientTape() as tape:
    predictions = f(x)
    loss = compute_loss(y, predictions)
    dloss_dw, dloss_db = tape.gradient(loss, [w, b])
  w.assign_sub(learning_rate * dloss_dw)
  b.assign_sub(learning_rate * dloss_db)
  return loss

"""
The code has not changed, but if you were running this code on GPU and on larger dataset - you would have noticed the difference in speed. 

## Dataset API

Tensorflow contains a convenient API to work with data. Let's try to use it. We will also train our model from scratch.
"""
logger.info("## Dataset API")

w.assign([[10.0]])
b.assign([0.0])

dataset = tf.data.Dataset.from_tensor_slices((train_x.astype(np.float32), train_labels.astype(np.float32)))
dataset = dataset.shuffle(buffer_size=1024).batch(256)

for epoch in range(10):
  for step, (x, y) in enumerate(dataset):
    loss = train_on_batch(tf.reshape(x,(-1,1)), tf.reshape(y,(-1,1)))
  logger.debug('Epoch %d: last batch loss = %.4f' % (epoch, float(loss)))

"""
## Example 2: Classification

Now we will consider binary classification problem. A good example of such a problem would be a tumour classification between malignant and benign based on it's size and age.

The core model is similar to regression, but we need to use different loss function. Let's start by generating sample data:
"""
logger.info("## Example 2: Classification")

np.random.seed(0) # pick the seed for reproducibility - change it to explore the effects of random variations

n = 100
X, Y = make_classification(n_samples = n, n_features=2,
                           n_redundant=0, n_informative=2, flip_y=0.05,class_sep=1.5)
X = X.astype(np.float32)
Y = Y.astype(np.int32)

split = [ 70*n//100, (15+70)*n//100 ]
train_x, valid_x, test_x = np.split(X, split)
train_labels, valid_labels, test_labels = np.split(Y, split)

def plot_dataset(features, labels, W=None, b=None):
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('$x_i[0]$ -- (feature 1)')
    ax.set_ylabel('$x_i[1]$ -- (feature 2)')
    colors = ['r' if l else 'b' for l in labels]
    ax.scatter(features[:, 0], features[:, 1], marker='o', c=colors, s=100, alpha = 0.5)
    if W is not None:
        min_x = min(features[:,0])
        max_x = max(features[:,1])
        min_y = min(features[:,1])*(1-.1)
        max_y = max(features[:,1])*(1+.1)
        cx = np.array([min_x,max_x],dtype=np.float32)
        cy = (0.5-W[0]*cx-b)/W[1]
        ax.plot(cx,cy,'g')
        ax.set_ylim(min_y,max_y)
    fig.show()

plot_dataset(train_x, train_labels)

"""
## Normalizing Data

Before training, it is common to bring our input features to the standard range of [0,1] (or [-1,1]). The exact reasons for that we will discuss later in the course, but in short the reason is the following. We want to avoid values that flow through our network getting too big or too small, and we normally agree to keep all values in the small range close to 0. Thus we initialize the weights with small random numbers, and we keep signals in the same range.

When normalizing data, we need to subtract min value and divide by range. We compute min value and range using training data, and then normalize test/validation dataset using the same min/range values from the training set. This is because in real life we will only know the training set, and not all incoming new values that the network would be asked to predict. Occasionally, the new value may fall out of the [0,1] range, but that's not crucial.
"""
logger.info("## Normalizing Data")

train_x_norm = (train_x-np.min(train_x)) / (np.max(train_x)-np.min(train_x))
valid_x_norm = (valid_x-np.min(train_x)) / (np.max(train_x)-np.min(train_x))
test_x_norm = (test_x-np.min(train_x)) / (np.max(train_x)-np.min(train_x))

"""
## Training One-Layer Perceptron

Let's use Tensorflow gradient computing machinery to train one-layer perceptron.

Our neural network will have 2 inputs and 1 output. The weight matrix $W$ will have size $2\times1$, and bias vector $b$ -- $1$.

Core model will be the same as in previous example, but loss function will be a logistic loss. To apply logistic loss, we need to get the value of **probability** as the output of our network, i.e. we need to bring the output $z$ to the range [0,1] using `sigmoid` activation function: $p=\sigma(z)$.

If we get the probability $p_i$ for the i-th input value corresponding to the actual class $y_i\in\{0,1\}$, we compute the loss as $\mathcal{L_i}=-(y_i\log p_i + (1-y_i)log(1-p_i))$. 

In Tensorflow, both those steps (applying sigmoid and then logistic loss) can be done using one call to `sigmoid_cross_entropy_with_logits` function. Since we are training our network in minibatches, we need to average out the loss across all elements of a minibatch using `reduce_mean`:
"""
logger.info("## Training One-Layer Perceptron")

W = tf.Variable(tf.random.normal(shape=(2,1)),dtype=tf.float32)
b = tf.Variable(tf.zeros(shape=(1,),dtype=tf.float32))

learning_rate = 0.1

@tf.function
def train_on_batch(x, y):
  with tf.GradientTape() as tape:
    z = tf.matmul(x, W) + b
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=z))
  dloss_dw, dloss_db = tape.gradient(loss, [W, b])
  W.assign_sub(learning_rate * dloss_dw)
  b.assign_sub(learning_rate * dloss_db)
  return loss

"""
We will use minibatches of 16 elements, and do a few epochs of training:
"""
logger.info("We will use minibatches of 16 elements, and do a few epochs of training:")

dataset = tf.data.Dataset.from_tensor_slices((train_x_norm.astype(np.float32), train_labels.astype(np.float32)))
dataset = dataset.shuffle(128).batch(2)

for epoch in range(10):
  for step, (x, y) in enumerate(dataset):
    loss = train_on_batch(x, tf.expand_dims(y,1))
  logger.debug('Epoch %d: last batch loss = %.4f' % (epoch, float(loss)))

"""
To make sure our training worked, let's plot the line that separates two classes. Separation line is defined by the equation $W\times x + b = 0.5$
"""
logger.info("To make sure our training worked, let's plot the line that separates two classes. Separation line is defined by the equation $W\times x + b = 0.5$")

plot_dataset(train_x,train_labels,W.numpy(),b.numpy())

"""
Let's see how our model behaves on the validation data.
"""
logger.info("Let's see how our model behaves on the validation data.")

pred = tf.matmul(test_x,W)+b
fig,ax = plt.subplots(1,2)
ax[0].scatter(test_x[:,0],test_x[:,1],c=pred[:,0]>0.5)
ax[1].scatter(test_x[:,0],test_x[:,1],c=valid_labels)

"""
To compute the accuracy on the validation data, we can cast boolean type to float, and compute the mean:
"""
logger.info("To compute the accuracy on the validation data, we can cast boolean type to float, and compute the mean:")

tf.reduce_mean(tf.cast(((pred[0]>0.5)==test_labels),tf.float32))

"""
Let's explain what goes on here:
* `pred` is the values predicted by the network. They are not quite probabilities, because we have not used an activation function, but values greater than 0.5 correspond to class 1, and smaller - to class 0.
*  `pred[0]>0.5` creates a boolean tensor of results, where `True` corresponds to class 1, and `False` - to class 0
* We compare that tensor to expected labels `valid_labels`, getting the boolean vector or correct predictions, where `True` corresponds to the correct prediction, and `False` - to incorrect one.
* We convert that tensor to floating point using `tf.cast`
* We then compute the mean value using `tf.reduce_mean` - that is exactly our desired accuracy

## Using TensorFlow/Keras Optimizers

Tensorflow is closely integrated with Keras, which contains a lot of useful functionality. For example, we can use different **optimization algorithms**. Let's do that, and also print obtained accuracy during training.
"""
logger.info("## Using TensorFlow/Keras Optimizers")

optimizer = tf.keras.optimizers.Adam(0.01)

W = tf.Variable(tf.random.normal(shape=(2,1)))
b = tf.Variable(tf.zeros(shape=(1,),dtype=tf.float32))

@tf.function
def train_on_batch(x, y):
  vars = [W, b]
  with tf.GradientTape() as tape:
    z = tf.sigmoid(tf.matmul(x, W) + b)
    loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(z,y))
    correct_prediction = tf.equal(tf.round(y), tf.round(z))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    grads = tape.gradient(loss, vars)
    optimizer.apply_gradients(zip(grads,vars))
  return loss,acc

for epoch in range(20):
  for step, (x, y) in enumerate(dataset):
    loss,acc = train_on_batch(tf.reshape(x,(-1,2)), tf.reshape(y,(-1,1)))
  logger.debug('Epoch %d: last batch loss = %.4f, acc = %.4f' % (epoch, float(loss),acc))

"""
**Task 1**: Plot the graphs of loss function and accuracy on training and validation data during training

**Task 2**: Try to solve MNIST classificiation problem using this code. Hint: use `softmax_crossentropy_with_logits` or `sparse_softmax_cross_entropy_with_logits` as loss function. In the first case you need to feed expected output values in *one hot encoding*, and in the second case - as integer class number.

## Keras
### Deep Learning for Humans

* Keras is a library originally developed by Francois Chollet to work on top of Tensorflow, CNTK and Theano, to unify all lower-level frameworks. You can still install Keras as a separate library, but it is not advised to do so. 
* Now Keras is included as part of Tensorflow library
* You can easily construct neural networks from layers
* Contains `fit` function to do all training, plus a lot of functions to work with typical data (pictures, text, etc.)
* A lot of samples
* Functional API vs. Sequential API

Keras provides higher level abstractions for neural networks, allowing us to operate in terms of layers, models and optimizers, and not in terms of tensors and gradients.  

Classical Deep Learning book from the creator of Keras: [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python)

### Functional API

When using functional API, we define the **input** to the network as `keras.Input`, and then compute the **output** by passing it through a series of computations. Finally, we define **model** as an object that transforms input into output.

Once we obtained **model** object, we need to:
* **Compile it**, by specifying loss function and the optimizer that we want to use with our model
* **Train it** by calling `fit` function with the training (and possibly validation) data
"""
logger.info("## Keras")

inputs = tf.keras.Input(shape=(2,))
z = tf.keras.layers.Dense(1,kernel_initializer='glorot_uniform',activation='sigmoid')(inputs)
model = tf.keras.models.Model(inputs,z)

model.compile(tf.keras.optimizers.Adam(0.1),'binary_crossentropy',['accuracy'])
model.summary()
h = model.fit(train_x_norm,train_labels,batch_size=8,epochs=15)

plt.plot(h.history['accuracy'])

"""
### Sequential API

Alternatively, we can start thinking of a model as of a **sequence of layers**, and just specify those layers by adding them to the `model` object:
"""
logger.info("### Sequential API")

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(5,activation='sigmoid',input_shape=(2,)))
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

model.compile(tf.keras.optimizers.Adam(0.1),'binary_crossentropy',['accuracy'])
model.summary()
model.fit(train_x_norm,train_labels,validation_data=(test_x_norm,test_labels),batch_size=8,epochs=15)

"""
## Classification Loss Functions

It is important to correctly specify loss function and activation function on the last layer of the network. The main rules are the following:
* If the network has one output (**binary classification**), we use **sigmoid** activation function, for **multiclass classification** - **softmax**
* If the output class is represented as one-hot-encoding, the loss function will be **cross entropy loss** (categorical cross-entropy), if the output contains class number - **sparse categorical cross-entropy**.  For **binary classification** - use **binary cross-entropy** (same as **log loss**)
* **Multi-label classification** is when we can have an object belonging to several classes at the same time. In this case, we need to encode labels using one-hot encoding, and use **sigmoid** as activation function, so that each class probability is between 0 and 1.

| Classification | Label Format | Activation Function | Loss |
|---------------|-----------------------|-----------------|----------|
| Binary      | Probability of 1st class | sigmoid | binary crossentropy |
| Binary      | One-hot encoding (2 outputs) | softmax | categorical crossentropy |
| Multiclass |  One-hot encoding | softmax | categorical crossentropy |
| Multiclass | Class Number | softmax | sparse categorical crossentropy |
| Multilabel | One-hot encoding | sigmoid | categorical crossentropy |

> Binary classification can also be handled as a special case of multi-class classification with two outputs. In this case, we need to use **softmax**.

**Task 3**: 
Use Keras to train MNIST classifier:
* Notice that Keras contains some standard datasets, including MNIST. To use MNIST from Keras, you only need a couple of lines of code (more information [here](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist))
* Try several network configuration, with different number of layers/neurons, activation functions.

What is the best accuracy you were able to achieve?

## Takeaways

* Tensorflow allows you to operate on tensors at low level, you have most flexibility.
* There are convenient tools to work with data (`td.Data`) and layers (`tf.layers`)
* For beginners/typical tasks, it is recommended to use **Keras**, which allows to construct networks from layers
* If non-standard architecture is needed, you can implement your own Keras layer, and then use it in Keras models
* It is a good idea to look at PyTorch as well and compare approaches. 

A good sample notebook from the creator of Keras on Keras and Tensorflow 2.0 can be found [here](https://t.co/k694J95PI8).
"""
logger.info("## Classification Loss Functions")

logger.info("\n\n[DONE]", bright=True)