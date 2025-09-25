from PIL import Image
from jet.logger import logger
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
# Classification of Pet's Real-Life Images

Lab Assignment from [AI for Beginners Curriculum](https://github.com/microsoft/ai-for-beginners).

Now it's time to deal with more challenging task - classification of the original [Oxford-IIIT Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/). Let's start by loading and visualizing the dataset.
"""
logger.info("# Classification of Pet's Real-Life Images")

# !wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
# !tar xfz images.tar.gz
# !rm images.tar.gz

"""
We will define generic function to display a series of images from a list:
"""
logger.info("We will define generic function to display a series of images from a list:")


def display_images(l,titles=None,fontsize=12):
    n=len(l)
    fig,ax = plt.subplots(1,n)
    for i,im in enumerate(l):
        ax[i].imshow(im)
        ax[i].axis('off')
        if titles is not None:
            ax[i].set_title(titles[i],fontsize=fontsize)
    fig.set_size_inches(fig.get_size_inches()*n)
    plt.tight_layout()
    plt.show()

"""
You can see that all images are located in one directory called `images`, and their name contains the name of the class (breed):
"""
logger.info("You can see that all images are located in one directory called `images`, and their name contains the name of the class (breed):")

fnames = os.listdir('images')[:5]
display_images([Image.open(os.path.join('images',x)) for x in fnames],titles=fnames,fontsize=30)

"""
To simplify classification and use the same approach to loading images as in the previous part, let's sort all images into corresponding directories:
"""
logger.info("To simplify classification and use the same approach to loading images as in the previous part, let's sort all images into corresponding directories:")

for fn in os.listdir('images'):
    cls = fn[:fn.rfind('_')].lower()
    os.makedirs(os.path.join('images',cls),exist_ok=True)
    os.replace(os.path.join('images',fn),os.path.join('images',cls,fn))

"""
Let's also define the number of classes in our dataset:
"""
logger.info("Let's also define the number of classes in our dataset:")

num_classes = len(os.listdir('images'))
num_classes

"""
## Preparing dataset for Deep Learning

To start training our neural network, we need to convert all images to tensors, and also create tensors corresponding to labels (class numbers). Most neural network frameworks contain simple tools for dealing with images:
* In Tensorflow, use `tf.keras.preprocessing.image_dataset_from_directory`
* In PyTorch, use `torchvision.datasets.ImageFolder`

As you have seen from the pictures above, all of them are close to square image ratio, so we need to resize all images to square size. Also, we can organize images in minibatches.
"""
logger.info("## Preparing dataset for Deep Learning")



"""
Now we need to separate dataset into train and test portions:
"""
logger.info("Now we need to separate dataset into train and test portions:")



"""
Now define data loaders:
"""
logger.info("Now define data loaders:")




"""
## Define a neural network

For image classification, you should probably define a convolutional neural network with several layers. What to keep an eye for:
* Keep in mind the pyramid architecture, i.e. number of filters should increase as you go deeper
* Do not forget activation functions between layers (ReLU) and Max Pooling
* Final classifier can be with or without hidden layers, but the number of output neurons should be equal to number of classes.

An important thing is to get the activation function on the last layer + loss function right:
* In Tensorflow, you can use `softmax` as the activation, and `sparse_categorical_crossentropy` as loss. The difference between sparse categorical cross-entropy and non-sparse one is that the former expects output as the number of class, and not as one-hot vector.
* In PyTorch, you can have the final layer without activation function, and use `CrossEntropyLoss` loss function. This function applies softmax automatically. 

> **Hint:** In PyTorch, you can use `LazyLinear` layer instead of `Linear`, in order to avoid computing the number of inputs. It only requires one `n_out` parameter, which is number of neurons in the layer, and the dimension of input data is picked up automatically upon first `forward` pass.
"""
logger.info("## Define a neural network")



"""
## Train the Neural Network

Now we are ready to train the neural network. During training, please collect accuracy on train and test data on each epoch, and then plot the accuracy to see if there is overfitting.
"""
logger.info("## Train the Neural Network")




"""
Even if you have done everything correctly, you will probably see that the accuracy is quite low.

## Transfer Learning

To improve the accuracy, let's use pre-trained neural network as feature extractor. Feel free to experiment with VGG-16/VGG-19 models, ResNet50, etc.

> Since this training is slower, you may start with training the model for the small number of epochs, eg. 3. You can always resume training to further improve accuracy if needed.

We need to normalize our data differently for transfer learning, thus we will reload the dataset again using different set of transforms:
"""
logger.info("## Transfer Learning")



"""
Let's load the pre-trained network:
"""
logger.info("Let's load the pre-trained network:")



"""
Now define the classification model for your problem:
* In PyTorch, there is a slot called `classifier`, which you can replace with your own classifier for the desired number of classes.
* In TensorFlow, use VGG network as feature extractor, and build a `Sequential` model with VGG as first layer, and your own classifier on top
"""
logger.info("Now define the classification model for your problem:")



"""
Make sure to set all parameters of VGG feature extractor not to be trainable
"""
logger.info("Make sure to set all parameters of VGG feature extractor not to be trainable")



"""
Now we can start the training. Be very patient, as training takes a long time, and our train function is not designed to print anything before the end of the epoch.
"""
logger.info("Now we can start the training. Be very patient, as training takes a long time, and our train function is not designed to print anything before the end of the epoch.")



"""
It seems much better now!

## Optional: Calculate Top 3 Accuracy

We can also computer Top 3 accuracy using the same code as in the previous exercise.
"""
logger.info("## Optional: Calculate Top 3 Accuracy")


logger.info("\n\n[DONE]", bright=True)