import os
import csv
import cv2
import matplotlib.pyplot as plt

path = '../data_p/'
# path = 'C:/Users/Qiang/Downloads/data_p/'

os.chdir(path)
samples = []

PATHS = ['track1_central/driving_log.csv',
		   'track1_recovery/driving_log.csv',
		   'track1_reverse/driving_log.csv',
		   'track1_recovery_reverse/driving_log.csv',
		   'track2_central/driving_log.csv',
			'track1_test/driving_log.csv',
            'track2_test/driving_log.csv',
		   'udacity/driving_log.csv']
# PATHS = [ 'udacity/driving_log.csv']
# PATHS = [ 'udacity/driving_log3.csv']

# PATHS = ['track1_central/driving_log.csv',
# 		   'track1_recovery/driving_log.csv',
# 		   'track1_reverse/driving_log.csv',
# 		   'track1_recovery_reverse/driving_log.csv',
# 		   'track2_central/driving_log.csv',
# 			'track1_test/driving_log.csv',
#             'track2_test/driving_log.csv']
# the first line is the column names.
for PATH in PATHS:
	with open(PATH) as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			samples.append(line)
images = []
angles = []
for sample in samples:
	source_path = sample[0] # center image
	tokens = source_path.split('/')
	filename = tokens[-1]
	if len(tokens) <2:
		continue
	elif len(tokens) == 2:
		local_path = './udacity/IMG/' + filename
	else:
		local_path = tokens[-3] + '/' + tokens[-2] + '/' + filename
	# print(local_path)
	image = cv2.imread(local_path)
	images.append(image)
	angle = sample[3]
	angles.append(angle)
	# plt.imshow(image)
	# plt.show()
	# print(len(image))
	# exit()

print('Number of images read:',len(images))
print(len(angles))

import numpy as np
X_train = np.array(images)
y_train = np.array(angles)

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

input_shape = (160, 320, 3)
model = Sequential()
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

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch = 3, batch_size = 32)


# save the model
os.chdir('.')
model.save('model.h5')
print ('saved')
