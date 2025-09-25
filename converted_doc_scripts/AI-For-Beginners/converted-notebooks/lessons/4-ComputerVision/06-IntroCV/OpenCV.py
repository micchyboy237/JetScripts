from jet.logger import logger
import cv2
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
# Computer Vision and OpenCV

This notebook is a part of [AI for Beginners Curriculum](http://aka.ms/ai-beginners).

[OpenCV](https://opencv.org/) is considered to be *de facto* standard for image processing. It contains a lot of useful algorithms, implemented in C++. You can call OpenCV from Python as well.

In this notebooks, we will give you some examples for using OpenCV. For more details, you can visit [Learn OpenCV](https://learnopencv.com/getting-started-with-opencv/) online course. 

First, let's `import cv2`, as well as some other useful libraries:
"""
logger.info("# Computer Vision and OpenCV")


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
## Loading Images

Images in Python can be conveniently represented by NumPy arrays. For example, grayscale image with size of 320x200 pixels would be stored in 200x320 array, and color image of the same dimension would have shape of 200x320x3 (for 3 color channels). 

Let's start by loading an image:
"""
logger.info("## Loading Images")

im = cv2.imread('data/braille.jpeg')
logger.debug(im.shape)
plt.imshow(im)

"""
As you can see, it is an image of braille text. Since we are not very interested in the actual color, we can convert it to black-and-white:
"""
logger.info("As you can see, it is an image of braille text. Since we are not very interested in the actual color, we can convert it to black-and-white:")

bw_im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
logger.debug(bw_im.shape)
plt.imshow(bw_im, cmap='gray')

"""
## Braille Image Processing

If we want to apply image classification to recognize the text, we need to cut out individual symbols to make them similar to MNIST images that we have seen before. This can be done using [object detection](../11-ObjectDetection/README.md) technique which we will discuss later, but also we can try to use pure computer vision for that. A good description of how computer vision can be used for character separation can be found [in this blog post](https://learnopencv.com/image-alignment-feature-based-using-opencv-c-python/) - we will only focus on some computer vision techniques here.

First, let's try to enhance the image a little bit. We can use the idea of **thresholding** (well described [in this OpenCV article](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html)):
"""
logger.info("## Braille Image Processing")

im = cv2.blur(bw_im,(3,3))
im = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                           cv2.THRESH_BINARY_INV, 5, 4)
im = cv2.medianBlur(im, 3)
_,im = cv2.threshold(im, 0, 255, cv2.THRESH_OTSU)
im = cv2.GaussianBlur(im, (3,3), 0)
_,im = cv2.threshold(im, 0, 255, cv2.THRESH_OTSU)
plt.imshow(im)

"""
To work with images, we need to "extract" individual dots, i.e. convert the images to a set of coordinates of individual dots. We can do that using **feature extraction** techniques, such as SIFT, SURF or [ORB](https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html):
"""
logger.info("To work with images, we need to "extract" individual dots, i.e. convert the images to a set of coordinates of individual dots. We can do that using **feature extraction** techniques, such as SIFT, SURF or [ORB](https://docs.opencv.org/4.x/d1/d89/tutorial_py_orb.html):")

orb = cv2.ORB_create(5000)
f,d = orb.detectAndCompute(im,None)
logger.debug(f"First 5 points: { [f[i].pt for i in range(5)]}")

"""
Let's plot all points to make sure we got things right:
"""
logger.info("Let's plot all points to make sure we got things right:")

def plot_dots(dots):
    img = np.zeros((250,500))
    for x in dots:
        cv2.circle(img,(int(x[0]),int(x[1])),3,(255,0,0))
    plt.imshow(img)

pts = [x.pt for x in f]
plot_dots(pts)

"""
To separate individual characters, we need to know the bounding box of the whole text. To find it out, we can just compute min and max coordinates:
"""
logger.info("To separate individual characters, we need to know the bounding box of the whole text. To find it out, we can just compute min and max coordinates:")

min_x, min_y, max_x, max_y = [int(f([z[i] for z in pts])) for f in (min,max) for i in (0,1)]
min_y+=13
plt.imshow(im[min_y:max_y,min_x:max_x])

"""
Also, this text can be partially rotated, and to make it perfectly aligned we need to do so-called **perspective transformation**. We will take a rectangle defined by points $(x_{min},y_{min}), (x_{min},y_{max}), (x_{max},y_{min}), (x_{max},y_{max})$ and align it with new image with proportional dimensions:
"""
logger.info("Also, this text can be partially rotated, and to make it perfectly aligned we need to do so-called **perspective transformation**. We will take a rectangle defined by points $(x_{min},y_{min}), (x_{min},y_{max}), (x_{max},y_{min}), (x_{max},y_{max})$ and align it with new image with proportional dimensions:")

off = 5
src_pts = np.array([(min_x-off,min_y-off),(min_x-off,max_y+off),
                    (max_x+off,min_y-off),(max_x+off,max_y+off)])
w = int(max_x-min_x+off*2)
h = int(max_y-min_y+off*2)
dst_pts = np.array([(0,0),(0,h),(w,0),(w,h)])
ho,m = cv2.findHomography(src_pts,dst_pts)
trim = cv2.warpPerspective(im,ho,(w,h))
plt.imshow(trim)

"""
After we get this well-aligned image, it should be relatively easy to slice it into pieces:
"""
logger.info("After we get this well-aligned image, it should be relatively easy to slice it into pieces:")

char_h = 36
char_w = 24
def slice(img):
    dy,dx = img.shape
    y = 0
    while y+char_h<dy:
        x=0
        while x+char_w<dx:
            if np.max(img[y:y+char_h,x:x+char_w])>0:
                yield img[y:y+char_h,x:x+char_w]
            x+=char_w
        y+=char_h

sliced = list(slice(trim))
display_images(sliced)

"""
You have seen that quite a lot of tasks can be done using pure image processing, without any artificial intelligence. If we can use computer vision techniques to make the work of a neural network simpler - we should definitely do it, because it will allow us to solve problems with smaller number of training data.

## Motion Detection using Frame Difference

Detecting motion on video stream is a very frequent task. For example, it allows us to get alerts when something happens on a surveillance camera. If we want to understand what's happening on the camera, we can then use a neural network - but it is much cheaper to use neural network when we know that something is going on.

The main idea of motion detection is simple. If the camera is fixed, then frames from the camera should be pretty similar to each other. Since frames are represented as arrays, just by subtracting those arrays for two subsequent frames we will get the pixel difference, which should be low for static frames, and become higher once there is substantial motion in the image.

We will start by learning how to open a video and convert it into a sequence of frames:
"""
logger.info("## Motion Detection using Frame Difference")

vid = cv2.VideoCapture('data/motionvideo.mp4')

c = 0
frames = []
while vid.isOpened():
    ret, frame = vid.read()
    if not ret:
        break
    frames.append(frame)
    c+=1
vid.release()
logger.debug(f"Total frames: {c}")
display_images(frames[::150])

"""
Since color is not so important for motion detection, we will convert all frames to grayscale. Then, we will compute the frame differences, and plot their norms to visually see the amount of activity going on:
"""
logger.info("Since color is not so important for motion detection, we will convert all frames to grayscale. Then, we will compute the frame differences, and plot their norms to visually see the amount of activity going on:")

bwframes = [cv2.cvtColor(x,cv2.COLOR_BGR2GRAY) for x in frames]
diffs = [(p2-p1) for p1,p2 in zip(bwframes[:-1],bwframes[1:])]
diff_amps = np.array([np.linalg.norm(x) for x in diffs])
plt.plot(diff_amps)
display_images(diffs[::150],titles=diff_amps[::150])

"""
Suppose we want to create a report that shows what happened in front of the camera by showing the suitable image each time something happens. To do it, we probably want to find out the start and end frame of an "event", and display the middle frame. To remove some noise, we will also smooth out the curve above with moving average function:
"""
logger.info("Suppose we want to create a report that shows what happened in front of the camera by showing the suitable image each time something happens. To do it, we probably want to find out the start and end frame of an "event", and display the middle frame. To remove some noise, we will also smooth out the curve above with moving average function:")

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

threshold = 13000

plt.plot(moving_average(diff_amps,10))
plt.axhline(y=threshold, color='r', linestyle='-')

"""
Now we can find out frames that have the amount of changes above the threshold by using `np.where`, and extract a sequence of consecutive frames that is longer than 30 frames:
"""
logger.info("Now we can find out frames that have the amount of changes above the threshold by using `np.where`, and extract a sequence of consecutive frames that is longer than 30 frames:")

active_frames = np.where(diff_amps>threshold)[0]

def subsequence(seq,min_length=30):
    ss = []
    for i,x in enumerate(seq[:-1]):
        ss.append(x)
        if x+1 != seq[i+1]:
            if len(ss)>min_length:
                return ss
            ss.clear()

sub = subsequence(active_frames)
logger.debug(sub)

"""
And finally we can display the image:
"""
logger.info("And finally we can display the image:")

plt.imshow(frames[(sub[0]+sub[-1])//2])

"""
You may notice that color scheme on this image does not look right! This is because OpenCV for historical reasons loads images in BGR color space, while matplotlib uses more traditional RGB color order. Most of the time, it makes sense to convert images to RGB immediately after loading them.
"""
logger.info("You may notice that color scheme on this image does not look right! This is because OpenCV for historical reasons loads images in BGR color space, while matplotlib uses more traditional RGB color order. Most of the time, it makes sense to convert images to RGB immediately after loading them.")

plt.imshow(cv2.cvtColor(frames[(sub[0]+sub[-1])//2],cv2.COLOR_BGR2RGB))

"""
## Extract Motion using Optical Flow

While just comparing two consecutive frames allows us to see the amount of changes, it does not give any information on what is actually moving and where. To get that information, there is a technique called **[optical flow](https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html)**:

* **Dense Optical Flow** computes the vector field that shows for each pixel where is it moving
* **Sparse Optical Flow** is based on taking some distinctive features in the image (eg. edges), and building their trajectory from frame to frame.

Read more on optical flow [in this great tutorial](https://learnopencv.com/optical-flow-in-opencv/).

Let's compute dense optical flow between our frames:
"""
logger.info("## Extract Motion using Optical Flow")

flows = [cv2.calcOpticalFlowFarneback(f1, f2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
         for f1,f2 in zip(bwframes[:-1],bwframes[1:])]
flows[0].shape

"""
As you can see, for each frame the flow has the dimension of the frame, and 2 channels, corresponding to x and y components of optical flow vector.

Displaying optical flow in 2D is a bit challenging, but we can use one clever idea. If we convert optical flow to polar coordinates, then we will get two components for each pixel: *direction* and *intensity*. We can represent intensity by the pixel intensity, and direction by different colors. We will create an image in [HSV (Hue-Saturation-Value) color space](https://en.wikipedia.org/wiki/HSV_color_space), where hue will be defined by direction, value - by intensity, and saturation will be 255.
"""
logger.info("As you can see, for each frame the flow has the dimension of the frame, and 2 channels, corresponding to x and y components of optical flow vector.")

def flow_to_hsv(flow):
    hsvImg = np.zeros((flow.shape[0],flow.shape[1],3),dtype=np.uint8)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsvImg[..., 0] = 0.5 * ang * 180 / np.pi
    hsvImg[..., 1] = 255
    hsvImg[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsvImg, cv2.COLOR_HSV2BGR)

start = sub[0]
stop = sub[-1]
logger.debug(start,stop)

frms = [flow_to_hsv(x) for x in flows[start:stop]]
display_images(frms[::25])

"""
So, in those frames greenish color corresponds to moving to the left, while blue - moving to the right.

Optical flow can be a great tool to draw conclusions about general direction of movement. For example, if you see that all pixels in a frame are moving in more or less one direction - you can conclude that there is camera movement, and try to compensate for that.
"""
logger.info("So, in those frames greenish color corresponds to moving to the left, while blue - moving to the right.")

logger.info("\n\n[DONE]", bright=True)