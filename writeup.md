Behavioral Cloning of Driving Simulator

Jorge Orbay


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/Center_image.jpg "Example Input"
[image2]: ./examples/Center_image_flipped.jpg "Example Flipped"
[image3]: ./examples/Center_image_flipped_cropped.jpg "Example Cropped"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md (this file) summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

python drive.py model.h5

and then starting up the simulator (which can be given on request)

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

	As said in model.py, my model is heavily based off of Nvidia's 
architecture used in End to End Learning for Self Driving Cars (https://arxiv.org/abs/1604.07316) . 

	The model starts with a normalization layer that makes all the datapoints between -.5 and .5. Then, the images are cropped so that the bottom (which for most images, is composed  of the hood of the car) and the top (the sky, which matters little for turning) are removed.

	The convolutional layers start with one 5x5 kernel layer with 24 filters. This layer has a stride of (2,2). Two more layers follow that are nearly equivalent except that they use 36 and 48 filters respectively.

	After the previous layers, four identical convolutional layers of 3x3 kernels, 64 filters, and strides of (1,1) are used one right after the other.

	All of the previous convolutional layers were followed  by relu activational layers.

	Previously, both maxpooling layers and dropout layers were used, but both were surprisingly found to worsen the model's performance.

	Finally, a flattening layer is  used, followed by 4 dense (fully-connected layers) layers of size 1164, 100, 50, and 10. They were all followed by relu activations. The last layer is fully connected to a single node layer, which provides the output.

####2. Attempts to reduce overfitting in the model


The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

The model was also trained on flipped versions of the datasets (with flipped turning orientations) to make sure that it did not overfit with left turns (which were the most common turns in the training sets because the training was done on a left turning track).

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. However, I did attempt to tune the epsilon value of Adam (which was  1e-8 by default) to try to improve the model's ability of escaping local minimums. All variations of  the default epsilon value were found to decrease the model's performance.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving and recovery driving (driving that entails steering away from facing the side of the road). I also used driving data provided  by Udacity.

###Model Architecture and Training Strategy

####1. Solution Design Approach

My first attempt at making a network was heavily based on the VGG network (you can learn of it at https://arxiv.org/pdf/1409.1556.pdf). It's a personal favorite and worked well for me for a previous udacity project (Sign Classifier, which can be found on my github). I first  used it with a lot of the recommendations provided by Udacity, such as image flipping, normalization, and use of the Adam optimizer. However, no matter how much I trained it or increased the number of convolutional layers (I even once tried training it for 10 epochs, which took about an hour on AWS's GPU servers), it kept crashing into a particular turn right after the bridge on my test track. I originally trained the network on a recording of myself driving 2 laps about the track and a recording given by Udacity. I tried to improve the model by recording myself on the particular turn in question, copying the recording 10x, and giving it to the network for training. Oddly, the network only performed worse. I assume I was overtraining it on a particular approach to the turn. I also removed any non-turning data from the recording from Udacity (hoping the network would be more likely to turn), but that only helped a little bit.

After going through my checklist of possibilities for improving my VGG based network, I decided to try a totally new network, which ended up being the CNN given in Nvidia's End to End Learning for Self Driving Cars paper (https://arxiv.org/abs/1604.07316). It immediately improved performance, and it even took care of a previous problem in which the car would repeatedly accelerate to 15km/h, and then deecelerate to 5 km/h. However, the network was still unable to pass the turn after the bridge. Only after I removed the training data from my recording of the bridge multiplied by 10 did the network perfectly pass the whole track.

During design, before switching to the Nvidia CNN, I also tried adjusting the epsilon value of the Adam Optimizer and the number of epochs. Both adjustments neglected to improve the results.

####2. Final Model Architecture

	The model was described in the first part of Model Architecture and Training Strategy. 


####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. An example image from the recordings can be found below. 

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover well. This was stored in a second folder  and also used.

To augment the data sat, I flipped images and angles to reduce overfitting on turns that only required turning left (which was the majority of  turns). An example can be seen below.

![alt text][image2]

After the collection process, I had 50,842 pictures. After flipping, this became 101,684 data pictures. They were further processed in pre-processing, which used normalization to make all pixel values centered on 0 with a range from -.5 to .5. The pictures also had the top 30 pixels and bottom 20 pixels cropped out, which were seen as extraneous because they only captured the sky and hood of the car. An example can be seen below.

![alt text][image3]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 because the validation error did not decrease at 4 or above. 

I used an adam optimizer so that manually training the learning rate wasn't necessary.

If you would like to see a demonstration of the network at work, please see the video.mp4.

