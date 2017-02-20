import os
import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np

path = '../data_p/'
path = 'C:/Users/Qiang/Downloads/data_p/'

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
PATHS = [ 'track1_recovery/driving_log.csv',
	'udacity/driving_log.csv']
PATHS = [ 'udacity/driving_log.csv',
			'track1_bridge/driving_log.csv'		 ]

PATHS = [#'track1_central/driving_log.csv',
# 		   'track1_recovery/driving_log.csv',
# 		   'track1_reverse/driving_log.csv',
# 		   'track1_recovery_reverse/driving_log.csv',
# 			'track1_test/driving_log.csv',

			# 'track1_bridge/driving_log.csv'	,
		 	# 'track1_bridge2/driving_log.csv',
		 	'udacity/driving_log.csv' ]
PATHS = ['track1_central/driving_log.csv',
		   'track1_recovery/driving_log.csv',
		   'track1_reverse/driving_log.csv',
		   'track1_recovery_reverse/driving_log.csv',
			'track1_test/driving_log.csv',

			# 'track1_bridge/driving_log.csv'	,
		 	# 'track1_bridge2/driving_log.csv',
		 	# 'udacity/driving_log.csv'
]
# the first line is the column names.
for PATH in PATHS:
	with open(PATH) as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
			samples.append(line)
images = []
angles = []
for sample in samples:
	# use left and right images
	# I tried, no improvements.
	# for i in range(3):
	i = 0
	source_path = sample[i] # center image
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

	# # #augment data, so the batch becomes 64.
	# image_flipped = np.fliplr(image)
	# angle_flipped = -1 * angle
	# images.append(image_flipped)
	# angles.append(angle_flipped)

print('Number of images read:',len(images))
print(len(angles))

X_train = np.asarray(images)
y_train = np.asarray(angles)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization

input_shape = (160, 320, 3)
model = Sequential()
# simple model
model.add(Lambda(lambda x:x / 255.0 - 0.5, input_shape=input_shape))
# model.add(BatchNormalization())
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(16,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(64))
model.add(Dense(1))

# another model
# model.add(Lambda(lambda x:x / 255.0 - 0.5, input_shape=input_shape))
# model.add(Cropping2D(cropping=((70,25),(0,0))))
# model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
# model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
# model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
# model.add(Convolution2D(64,5,5,subsample=(2,2),activation='relu'))
# model.add(Convolution2D(64,5,5,subsample=(2,2),activation='relu'))
# model.add(Flatten())
# model.add(Dense(100))
# model.add(Dense(50))
# model.add(Dense(10))
# model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch = 5, batch_size = 32)


# save the model
os.chdir('.')
model.save('model.h5')
print ('saved')
