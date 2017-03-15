import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

#This series of blocks creates lines, an array of image paths that each direct to a separate image obtained when recording users driving the 
#car simulator in a manner I deem acceptable. Three different instances of training were used, and the filepaths of the images were stored in
#three separate csv's.
#The fourth with block that was commented out represents an attempt to force the network to learn how to deal with a particularly tough turn.
#This attempt was later show to actually hurt the network because I was giving it too many copies of the same pictures.
lines = []
with open('../recorded_data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

with open('../recorded_data_2/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)
with open('../data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)
#with open('../recorded_data_4/driving_log.csv') as csvfile:
#	reader = csv.reader(csvfile)
#	for line in reader:
#		for i in range (0,10):
#			lines.append(line)




#The next few lines are a setup for the generator, which allows a user
#to import picture matrices per batch, rather than all at once, which
#would use up much more memory. The data is shuffled befofre splitting it into batches
import os
from sklearn.model_selection import train_test_split
import sklearn

#This splits the training (4/5 of the data) and validation (1/5 of the data) sets
train_samples, validation_samples = train_test_split(lines,test_size=.2)

def generator(samples, batch_size = 32):
	num_samples = len(samples)

	#the dataTesters are only used to ascertain which data sets are being used
	
	dataTester01 = 0
	dataTester02 = 0
	dataTester03 = 0
	dataTester04 = 0

	#This loop is run for the duration of the process
	while 1:
		random.shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]
			images = []
			angles = []

	
			
			for batch_sample in batch_samples:
			
				#This series of if statements is used to change the datapaths from the csv's into usable datapaths
				#for the current system
	
				#the next if statement is done to distinguish the one dataset made on a windows computer

				if (len(batch_sample[0].split('\\')) < 2):
					if (len(batch_sample[0].split('/')) < 3):

						name = '../data/IMG/' + batch_sample[0].split('/')[-1]

						if dataTester01==0:
							#print('dataTester01')
							dataTester01 = dataTester01+1

					elif (batch_sample[0].split('/')[-3] == 'recorded_data_4'):

						name = '../recorded_data_4/IMG/' + batch_sample[0].split('/')[-1]
	
						if dataTester02==0:
							#print('dataTester02')
							dataTester02 = dataTester02+1

					else:

						name = '../recorded_data_2/IMG/' + batch_sample[0].split('/')[-1]
						
						
						if dataTester03==0:
							#print('dataTester03')
							dataTester03 = dataTester03+1
				else:

					name = '../recorded_data/IMG/' + batch_sample[0].split('\\')[-1]

					
					if dataTester04==0:
						#print('dataTester04')
						dataTester04 = dataTester04+1

				#The next few lines obtain the particular image matrix (center_image) and the particular
				#steering angle (center_angle) and add them to arrays of image matrices and steering angles
				
				center_image = cv2.imread(name)
				center_angle = float(batch_sample[3])
				images.append(center_image)
				angles.append(center_angle)
		
			#The next line and for loop make horizontally flipped copies of every image and 
			#flip the side of the relevant steering angles to avoid overfitting for the particular dataset
			#which has a majority of left turning images because they were taken on a left turning track
			augmented_images, augmented_angles = [],[]

			for image,angle in zip(images,angles):
				augmented_images.append(image)
				augmented_angles.append(angle)
				augmented_images.append(cv2.flip(image,1))
				augmented_angles.append(angle*(-1.0))

			#These last lines return the currently needed batches and shuffle them. It will continue on the next iteration
			#of the offset for loop or, if the current epoch is done, on the next iteration of the while 1 loop
			X_train = np.array(augmented_images)
			y_train = np.array(augmented_angles)

			yield sklearn.utils.shuffle(X_train, y_train)

#instantiates a generator for the training and validation data sets
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import  MaxPooling2D
from keras.models import  Model
import keras


#print(X_train.shape)
#print(y_train.shape)

#gosh, Keras is lovely. It is so easy to use
#We will now build the network architecture, which is heavily based
#Nvidia's architecture from their End to End Learning for Self-Driving
#Cars paper (https://arxiv.org/abs/1604.07316)

#The only changes that seemed to improve the network were a cropping 
#layer, which removed the sky from the pictures, relu activation after
#every convolutional or dense layer (except the last), and two more 
#convolutional layers
model = Sequential()

model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))

#For some reason, the model could not crop until after the lambda layer
model.add(Cropping2D(cropping=((30,20),(1,1))))


#let's add convolution layers

model.add(Convolution2D(24,5,5,border_mode='valid', subsample = (2,2), activation = 'relu'))

model.add(Convolution2D(36,5,5,border_mode='valid', subsample = (2,2),activation='relu'))

#Pooling was found to hurt the model's performance
#model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(48,5,5,border_mode='valid', subsample = (2,2),activation='relu'))

#Dropout layers were also found to hurt the model's performance
#model.add(Dropout(.25))


model.add(Convolution2D(64,3,3,border_mode='valid',activation='relu'))

model.add(Convolution2D(64,3,3,border_mode='valid',activation='relu'))

#model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(64,3,3,border_mode='valid',activation='relu'))

model.add(Convolution2D(64,3,3,border_mode='valid',activation='relu'))


#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(.25))

#This is now just a series of dense  (fully-connected) layers. The 
#last layer is not followed  by relu activation because we do not 
# want only two possible steering outputs
model.add(Flatten())
model.add(Dense(1164,activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(1))

#The adam optimizer, as recommended by Udacity, performed well. 
#I tried adjusting its epsilon value so as to escape local minimum's 
#with more ease, but it only hurt performance
adamOpt = keras.optimizers.Adam()


#The rest of the code compiles the model and trains it for 3 epochs.
#Only three epochs were used to avoid overfitting on the training data.
model.compile(loss='mse', optimizer = adamOpt)


model.fit_generator(train_generator, samples_per_epoch = len(train_samples), validation_data=validation_generator, nb_val_samples = len(validation_samples), nb_epoch=3)

model.save('model.h5')

print("done building model.h5")
