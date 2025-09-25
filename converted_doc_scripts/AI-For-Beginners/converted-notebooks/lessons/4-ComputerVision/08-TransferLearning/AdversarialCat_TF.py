from IPython.display import clear_output
from PIL import Image
from jet.logger import logger
from tensorflow import keras
import json
import matplotlib.pyplot as plt
import numpy as np
import os
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
## How Neural Network sees a Cat

A neural network pre-trained on ImageNet is capable of recognizing any of 1000 different classes of objects, such as cats of different breeds. It would be interesting to see, what does the **ideal siamese cat** looks like for a neural network.

> Of course, you can replace *siamese cat* with any other ImageNet class.

To start, let's load VGG network:
"""
logger.info("## How Neural Network sees a Cat")

np.set_printoptions(precision=3,suppress=True)

model = keras.applications.VGG16(weights='imagenet',include_top=True)
classes = json.loads(open('imagenet_classes.json','r').read())

"""
## Optimizing for Result

To visualize the ideal cat, we will start with a random noise image, and will try to use the gradient descent optimization technique to adjust the image to make a network recognize a cat.

![Optimization Loop](images/ideal-cat-loop.png)

Here is our starting image:
"""
logger.info("## Optimizing for Result")

x = tf.Variable(tf.random.normal((1,224,224,3)))

def normalize(img):
    return (img-tf.reduce_min(img))/(tf.reduce_max(img)-tf.reduce_min(img))

plt.imshow(normalize(x[0]))

"""
> We use `normalize` function to bring our values into 0-1 range.

If we call our VGG network on this image, we will get more or less random distribution of probabilities:
"""
logger.info("If we call our VGG network on this image, we will get more or less random distribution of probabilities:")

def plot_result(x):
    res = model(x)[0]
    cls = tf.argmax(res)
    logger.debug(f"Predicted class: {cls} ({classes[cls]})")
    logger.debug(f"Probability of predicted class = {res[cls]}")
    fig,ax = plt.subplots(1,2,figsize=(15,2.5),gridspec_kw = { "width_ratios" : [1,5]} )
    ax[0].imshow(normalize(x[0]))
    ax[0].axis('off')
    ax[1].bar(range(1000),res,width=3)
    plt.show()

plot_result(x)

"""
> Even though it may look like the probability of one of the classes is much higher than the others, it is still very low - look at the scale to see that actual probability is still around 5%.

Now let's chose one target category (eg., **siamese cat**), and start adjusting the image using gradient descent. If $x$ is the input image, and $V$ is the VGG network, we will calculate the loss function $\mathcal{L} = \mathcal{L}(c,V(x))$ (where $c$ is the target category), and adjust $x$ using the following formula:
$$
x^{(i+1)} = x^{(i)} - \eta{\partial \mathcal{L}\over\partial x}
$$
Loss function would be cross-entropy loss, because we are comparing two probability distributions. In our case, because the class is represented by a number, and not by one-hot encoded vector, we will use *sparse categorical cross-entropy*.

We will repeat this process for several epochs, printing the image as we go.

> It is better to execute this code on GPU-enabled compute, or reduce the number of epochs in order to minimize waiting time.
"""
logger.info("Now let's chose one target category (eg., **siamese cat**), and start adjusting the image using gradient descent. If $x$ is the input image, and $V$ is the VGG network, we will calculate the loss function $\mathcal{L} = \mathcal{L}(c,V(x))$ (where $c$ is the target category), and adjust $x$ using the following formula:")

target = [284] # Siamese cat

def cross_entropy_loss(target,res):
    return tf.reduce_mean(keras.metrics.sparse_categorical_crossentropy(target,res))

def optimize(x,target,epochs=1000,show_every=None,loss_fn=cross_entropy_loss, eta=1.0):
    if show_every is None:
        show_every = epochs // 10
    for i in range(epochs):
        with tf.GradientTape() as t:
            res = model(x)
            loss = loss_fn(target,res)
            grads = t.gradient(loss,x)
            x.assign_sub(eta*grads)
            if i%show_every == 0:
                clear_output(wait=True)
                logger.debug(f"Epoch: {i}, loss: {loss}")
                plt.imshow(normalize(x[0]))
                plt.show()

optimize(x,target)

plot_result(x)

"""
We now have obtained an image that looks like a cat for a neural network, even though it still looks like a noise for us. If we optimize for a little bit longer - we are likeley to get the image of **ideal noisy cat**, which has probability close to 1.

## Making Sense of Noise

This noise does not make a lot of sense for us, but most probably it contains a lot of low-level filters that are typical for a cat. However, because there are very many ways to optimize input for the ideal result, the optimization algorithm is not motivated to find patterns that are visually comprehensible.

To make this look a little bit less like a noise, we can introduce an additional term to the loss function - **variation loss**. It measures how similar neighboring pixels of the image are. If we add this term to our *loss function*, it will force the optimizer to find solutions with less noise, and thus having more recognizable details.

> In practice, we need to balance between cross-entropy loss and variation loss to obtain good results. In our function, we introduce some numeric coefficients, and you can play with them and observe how image changes.
"""
logger.info("## Making Sense of Noise")

def total_loss(target,res):
    return 10*tf.reduce_mean(keras.metrics.sparse_categorical_crossentropy(target,res)) + \
           0.005*tf.image.total_variation(x,res)

optimize(x,target,loss_fn=total_loss)

"""
This is the ideal image of a cat for our neural network, and we can also see some of the familiar features, such as eyes and ears. There are many of them, which makes neural network even more certain that this is a cat.
"""
logger.info("This is the ideal image of a cat for our neural network, and we can also see some of the familiar features, such as eyes and ears. There are many of them, which makes neural network even more certain that this is a cat.")

plot_result(x)

"""
Let's also see how some other object looks like for the VGG:
"""
logger.info("Let's also see how some other object looks like for the VGG:")

x = tf.Variable(tf.random.normal((1,224,224,3)))
optimize(x,[340],loss_fn=total_loss) # zebra

"""
## Adversarial Attacks

Since *ideal cat* image can look like a random noise, it suggests that we can maybe tweak any image in a little way so that it changes it's class. Let's experiment with this a little bit. We will start with an image of a dog:
"""
logger.info("## Adversarial Attacks")

img = Image.open('images/dog-from-unsplash.jpg')
img = img.crop((200,20,600,420)).resize((224,224))
img = np.array(img)
plt.imshow(img)

"""
We can see that this image is clearly recognized as a dog:
"""
logger.info("We can see that this image is clearly recognized as a dog:")

plot_result(np.expand_dims(img,axis=0))

"""
Now, we will use this image a starting point, and try to optimize it to become a cat:
"""
logger.info("Now, we will use this image a starting point, and try to optimize it to become a cat:")

x = tf.Variable(np.expand_dims(img,axis=0).astype(np.float32)/255.0)
optimize(x,target,epochs=100)

plot_result(x)

"""
So, this image above is a perfect cat, from the point of view of VGG network!

## Experimenting with ResNet

Let's now see how this same image is classified by a different model, say, ResNet:
"""
logger.info("## Experimenting with ResNet")

model = keras.applications.ResNet50(weights='imagenet',include_top=True)

"""
> Since we used `model` as a global variable, from now on all functions will use ResNet instead of VGG
"""

plot_result(x)

"""
Apparenlty, the result is quite different. This is quite expected, because when optimizing for a cat we took into account the nature of VGG network, it's low-level filters, etc. Since ResNet has different filters, it gives different results. This gives us the idea of how we can protect ourselves from adversarial attacks - by using ensemble of different models.

Let's see how the ideal zebra looks like for ResNet:
"""
logger.info("Apparenlty, the result is quite different. This is quite expected, because when optimizing for a cat we took into account the nature of VGG network, it's low-level filters, etc. Since ResNet has different filters, it gives different results. This gives us the idea of how we can protect ourselves from adversarial attacks - by using ensemble of different models.")

x = tf.Variable(tf.random.normal((1,224,224,3)))
optimize(x,target=[340],epochs=500,loss_fn=total_loss)

plot_result(x)

"""
This picture is quite different, which tells us that the architecture of a neural network probably plays quite an important role in the way it recognizes objects.

> **Task**: Try to perform adversarial attach on ResNet, and compare the results.

## Using Different Optimizers

In our example, we have been using the simplest optimization technique - gradient descent. However, Keras framework contains different built-in [optimizers](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers), and we can use them instead of gradient descent. This will require very little change to our code - we will replace the part where we adjust input image `x.assign_sub(eta*grads)` with a call to `apply_gradients` function of the optimizer:
"""
logger.info("## Using Different Optimizers")

def optimize(x,target,epochs=1000,show_every=None,loss_fn=cross_entropy_loss,optimizer=keras.optimizers.SGD(learning_rate=1)):
    if show_every is None:
        show_every = epochs // 10
    for i in range(epochs):
        with tf.GradientTape() as t:
            res = model(x)
            loss = loss_fn(target,res)
            grads = t.gradient(loss,x)
            optimizer.apply_gradients([(grads,x)])
            if i%show_every == 0:
                clear_output(wait=True)
                logger.debug(f"Epoch: {i}, loss: {loss}")
                plt.imshow(normalize(x[0]))
                plt.show()

x = tf.Variable(tf.random.normal((1,224,224,3)))

optimize(x,[898],loss_fn=total_loss) # water bottle

"""
## Conclusion

We were able to visualize the ideal image of a cat (as well as any other objects) within pre-trained CNN, using gradient descent optimization to adjust the input image instead of weights. The main trick to get the image that makes some sense was to use variation loss as an additional loss function, which enforces the image to look smoother.
"""
logger.info("## Conclusion")

logger.info("\n\n[DONE]", bright=True)