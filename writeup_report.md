# **Behavioral Cloning** 
## An Implementation of Nvidia's Paper: **End-to-End Learning for Self Driving Cars**

---

**Behavioral Cloning Project**

This is my solution to the 3rd project in the udacity self driving car nanodegree program. The goal of this project is to create a CNN model that can predict the steering angle from images. The model architecture was based on Nvidia's paper ["End-to-End Learning for Self Driving Cars"](https://arxiv.org/abs/1604.07316). A PID controller was added to improve the performance and to control the speed/throttle of the car.

[//]: # (Image References)

[image1]: ./images/figure_1.png "Unbalanced Dataset"
[image2]: ./images/figure_2.png "Balanced Dataset"
[image3]: ./images/architecture.png "Nvidia Architecture"
[image4]: ./images/placeholder_small.png "Recovery Image"
[image5]: ./images/placeholder_small.png "Recovery Image"
[image6]: ./images/placeholder_small.png "Normal Image"
[image7]: ./images/placeholder_small.png "Flipped Image" 

---

My project includes the following files:
* [model.py] containing the script to create and train the model
* [utils.py] containing some helper functions and classes
* [drive.py] for driving the car in autonomous mode
* [model.h5 - model.json] containing a trained convolution neural network
* [video.mp4] containing a video of the car driving autonomously for two laps
* [writeup_report.md] summarizing the results

## How was it all done?

#### 1. Dataset Collection and Pre-processing
I first used the dataset provided by Udacity, which contained around 24k images, but because training needs a lot of data, I used the Udacity self driving car simulator. I collected around 37k images.

One of the biggest problems in dataset collection was that the data was unbalanced.

![alttext][image1]

I randomly picked some data points and removed them so that the dataset is balanced.

![alttext][image2]

The preprocessing step on each image consisted of the following:
* Cropping the image
* Resizing the image
* Converting the color space of the image from RGB to YUV

I also added random brightness, flipping, translation, and shadow to provide some sort of augmentation to the training batches.

#### 2. Model Architecture
The model architecture was based on Nvidia's [End-to-End Learning for Self Driving Cars](https://arxiv.org/abs/1604.07316). It consisted of 5 convolution layers (3 5x5 filters, and 2 3x3 filters), a dropout, and 3 dense layers before the output. I tried tanh as an activation function and it worked to some extint, but I used an ELU function because it worked the best. I also added l2 regularizers on every layer.

#### 3. Code Structure

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

utils.py contains all the functions that I used for preprocessing as well as the batch generation function. I also added a PID class for the use of controlling the speed and throttle of the car (I'll come to that in further detail later).

drive.py contains the code for opening a socket connection with the simulator and driving the car. You can run the program using the following command:

```python drive.py model.json```

model.h5, model.json, and video.mp4 contains a saved keras model, and a video of the car driving autonomously.

## Model Architecture and Training Strategy

### Solution Design Approach

I read several research papers on similar topics, and I found both the publications of Nvidia and comma.ai so interesting. Thus, I decided to build my model based on them. I also thought about trying image classification architectures (e.g. AlexNet or ImageNet) but did not have a good setup to try them.
 
I first tried to implement AlexNet, but it did not fit in my laptop's memory (and that was expected), so I started working on several variations of Nvidia's and comma.ai's architectures. I tried several activation functions (tanh, relu, and ELU), several filter sizes, several normalization layers, dropout values, and other things until getting to the final model.

I noticed that changing the architecture is really not enough to have a working result. While changing on the architecture, I changed a lot on the preprocessing steps and the random augmentation functions.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I had several results with varying losses and accuracies. I used the mean squared error as my loss function because it puts a high penalty on errors, and we want to minimize that.

To combat the overfitting, I added regularizers, a dropout and used a subsampling of size 2x2 on every convolution layer. I also used the augmentation functions to help the model better generalize.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, especially on corners and after the bridge. So, I tried to collect more data, and I lowered the learning rate until I got a working model.

#### 2. Final Model Architecture

The final model architecture (model.py lines 86-111) consisted of a convolution neural network with 5 convolution layers and 3 dense layers. The first three convolution layers had a filter size of 5x5, and the last two convolution layers had a filter size of 3x3.

I used an ELU activation function with an Adam optimizer, an MSE loss function, and a learning rate of 1e-4.

![alt text][image3]

## What is next?

#### New method for dataset collection
Mr. Eric advised me on slack with an interesting method of collecting data. When I first collected the dataset, I reached around 30k images. And after dataset balancing I reached ~7k data points. So Mr. Eric proposed a way to collect only 100 examples that would be sufficient to train a working model. Of course that would need some edits on the model. 

I am also planning to redo the whole project in a different way. I was thinking that I could use computer vision techniques and a PID or a Model Predictive Controller to control the car. This method will allow the car to drive autonomously in both tracks without any need for a dataset or for training it.
