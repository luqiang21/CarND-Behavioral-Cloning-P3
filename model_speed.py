import os
import csv
path = '../data_p/'
path = 'C:/Users/Qiang/Downloads/data_p/'
path = '/Users/LuQiang/Downloads/data_p'

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
PATHS = ['track1_central/driving_log.csv',
		   'track1_recovery/driving_log.csv',
		   'track1_reverse/driving_log.csv',
		   'track1_recovery_reverse/driving_log.csv',
			'track1_test/driving_log.csv',
		 	'track1_curve_after_bridge/driving_log.csv',
		   'udacity/driving_log.csv']
PATHS = [	'track1_curve_after_bridge/driving_log.csv',
		 	'udacity/driving_log.csv' ]
PATHS = [ 'udacity/driving_log.csv']
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

from sklearn.model_selection import train_test_split
# print(len(samples))
samples = samples[:(len(samples) // 32) * 32]
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
print(len(samples))
import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle
# ch, row, col = 3, 66, 200  # Nvidia's paper input size
ch, row, col = 3, 160, 320  #

def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		# shuffle the data
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			speeds = []
			for batch_sample in batch_samples:
				if len(batch_sample[0].split('/')) < 2:
					continue
				elif len(batch_sample[0].split('/')) == 2:
					name = 'udacity/IMG/'+batch_sample[0].split('/')[-1]
				else:
				# directory = batch_sample[i].split('/')[-2]
				# name = './'+directory+'/'+batch_sample[i].split('/')[-1]
					name = batch_sample[0].split('/')[-3] + '/IMG/'+batch_sample[0].split('/')[-1]


				center_image = cv2.imread(name)
				# trim image to only see section with road
				# print('name', name)
				# print('len of image ', len(center_image),'name:', name)
				if center_image == None:
					break
				shape = center_image.shape

				# center_image = center_image[int(shape[0]/3):shape[0], 0:shape[1]]
				# center_image = cv2.resize(center_image, (row, col), interpolation=cv2.INTER_AREA)
				# center_image = cv2.cvtColor(center_image, cv2.COLOR_RGB2YUV)
				# center_image = center_image.reshape(row, col, ch)

				speed = float(batch_sample[-1])
				images.append(center_image)
				speeds.append(speed)



			X_train = np.array(images)
			y_train = np.array(speeds)


			# print(X_train[0].shape)
			# print(len(X_train), len(y_train))
			yield sklearn.utils.shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, ELU, Cropping2D
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam

# Preprocess incoming data, centered around zero with small standard deviation


input_shape = (row, col, ch)
##### Simple version
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




##### Nvidia
# model = Sequential()
# # model.add(MaxPooling2D(pool_size=(2,3),input_shape=input_shape))
# # normalize to [-1, 1]
# # model.add(Lambda(lambda x: x/127.5 - 1.))#,
# 		# input_shape=(160, 320, 3),
# 		#output_shape=(row, col, ch)))
# model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = input_shape))
# model.add(Convolution2D(5, 5, 24, subsample=(4, 4), border_mode="same"))
# model.add(ELU())
# model.add(Convolution2D(5, 5, 36, subsample=(2, 2), border_mode="same"))
# model.add(ELU())
# model.add(Convolution2D(5, 5, 48, subsample=(2, 2), border_mode="same"))
# model.add(ELU())
# model.add(Convolution2D(3, 3, 64, subsample=(2, 2), border_mode="same"))
# model.add(ELU())
# model.add(Convolution2D(3, 3, 64, subsample=(2, 2), border_mode="same"))
# model.add(Flatten())
# model.add(Dropout(.2))
# model.add(ELU())
# model.add(Dense(1164))
# model.add(Dropout(.2))
# model.add(ELU())
# model.add(Dense(100))
# model.add(Dropout(.2))
# model.add(ELU())
# model.add(Dense(50))
# model.add(Dropout(.2))
# model.add(ELU())
# model.add(Dense(10))
# model.add(Dropout(.2))
# model.add(ELU())
# model.add(Dense(1))

# adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# model.compile(optimizer=adam, loss="mse", metrics=['accuracy'])
#
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=
			len(train_samples), validation_data=validation_generator,
			nb_val_samples=len(validation_samples), nb_epoch=5)
model.summary()

# save the model
os.chdir('.')
model.save('model.h5')
