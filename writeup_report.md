#**Behavioral Cloning** 


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image2]: ./images/center.jpg "center"
[image3]: ./images/recover1.jpg "Recovery Image"
[image4]: ./images/recover2.jpg "Recovery Image"
[image5]: ./images/recover3.jpg "Recovery Image"
[image6]: ./images/original.png "Normal Image"
[image7]: ./images/flipped.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths between 6 and 16 (model.py lines 133-135) 

The model includes RELU layers to introduce nonlinearity (code line 133-135), and the data is normalized in the model using a Keras lambda layer (code line 131). 


<!---
####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.
-->
####2. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 182).

####3. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. And left, right, center images are all used for traininig. 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to able to predict steering angle given a road image.

My first step was to use a convolution neural network model which includes max pooling layers and convolution layers that uses 'relu' as the activation function.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.

The final step was to run the simulator to see how well the car was driving around track one. The car was not able to cross the bridge or sometimes it was driving off the curve following the bridge. After seeking help from others, I included the left and right images in the data set and tuned the correction angle to be 15 degree (about 0.25 in radian). 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 130-140) consisted of a convolution neural network with the following layers and layer sizes. Normalization and cropping are also done here.

Here is the architecture.

```python
model.add(Lambda(lambda x:x / 255.0 - 0.5, input_shape=input_shape))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(16,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(64))
model.add(Dense(1))
```

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded several laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to drive to the lane's center if it drives off the lane's center. These images show what a recovery looks like starting from the right curb:

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would give more robust performance. For example, here is an image that has then been flipped:

 (Note: in the submission model, I commented the flipping part to save time since my AWS has some problem I was running the program on my local machine. And the data used is only the udacity data to save time.) 
![alt text][image6]
![alt text][image7]



After the collection process, I had 24108 number of data points. I then put 20% of the data into a validation set. I used the other 80% of the data set as training data for training the model. The validation set helped determine if the model was over or under fitting. I used the generator method to deal with this large data set. The generator could pull pieces of the data and process them on the fly only when I need them which is much more memory-efficient. The batch size for the generator is 32 and the data are shuffled before they are divided into batches. The normalization and cropping of the image are also done in the model part. The ideal number of epochs was 5 since 3 gave me poor performance and 7 didn't improve the performance much.  I used an adam optimizer so that manually training the learning rate wasn't necessary.
