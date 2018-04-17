import csv
import utils
import datetime
import glob

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
pd.options.display.float_format = '${:,.10f}'.format

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from scipy.misc import imread

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import BatchNormalization, Conv2D, ConvLSTM2D, Dense, MaxPooling2D, Dropout, Flatten, Lambda, Convolution2D
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2, activity_l2
from keras.callbacks import ModelCheckpoint
from keras.utils.vis_utils import plot_model

###### TRAINING CONSTANTS ######
SPLIT = 0.7
BATCH_SIZE = 40
EPOCHS = 30
SAMPLES_PER_EPOCH = (20000//BATCH_SIZE)*BATCH_SIZE
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
############################

# Reading from the clean csv after the path fixing
# Note that custom collected data is joined with
# the udacity data
base_path = '/Users/mohammedamarnah/Desktop/SDCProject/dataset/'
paths = glob.glob(base_path+'*')

app = False
for i in range(len(paths)):
	if paths[i][-1] == '2':
		continue
	utils.fixPath(paths[i]+'/IMG/', paths[i]+'/', append=app)
	app = True

data = pd.read_csv('../../dataset/data/driving_log_clean.csv')

# Shuffling the data
data = data.sample(frac=1).reset_index(drop=True)

# Reading the data from the pandas dataframe
X = data[['center', 'left', 'right']].values
y = data['steering'].values

# Balancing the dataset, and dropping some of the common examples
X, y = utils.balance_data(X, y)

# Some information about the data after balancing
print("Full Data Size: ", data.shape)
print("Data After Balancing: ", len(X)+len(y))

# Splitting the data: (See SPLIT under SOME CONSTANTS)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=SPLIT, random_state=0)

# Some information about the data after splitting
print("Splitting with split rate: ", SPLIT)
print("Training Data Size: ", X_train.shape, y_train.shape)
print("Validation Data Size: ", X_valid.shape, y_valid.shape)

# Freeing the memory block cause you know, it needs to be free.
data = None
X = None
y = None

ans = str(input("Continue? ([Y]/N) --- "))
if (ans == 'N' or ans == 'n'):
	exit()

# Generate training and validation data for the model compilation
train = utils.gen_batches(X_train, y_train, True, BATCH_SIZE)
valid = utils.gen_batches(X_valid, y_valid,  False, BATCH_SIZE)

######################### Model Training ###########################

def nvidia(LR=1e-4, inputshape=(64, 64, 1), comp=False, summary=False):
	model = Sequential()
	model.add(Lambda(lambda x: x/127.5-1.0, input_shape=inputshape))
	
	model.add(Conv2D(24, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
	model.add(ELU())
	model.add(Conv2D(36, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
	model.add(ELU())
	model.add(Conv2D(48, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
	model.add(ELU())
	model.add(Conv2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
	model.add(ELU())
	model.add(Conv2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
	model.add(ELU())
	
	model.add(Dropout(0.5))

	model.add(Flatten())
	model.add(Dense(100, W_regularizer=l2(0.001)))
	model.add(ELU())
	model.add(Dense(50,  W_regularizer=l2(0.001)))
	model.add(ELU())
	model.add(Dense(10,  W_regularizer=l2(0.001)))
	model.add(ELU())

	model.add(Dense(1))
	
	if comp:
		model.compile(optimizer=Adam(lr=LR), loss='mse', metrics=['accuracy'])
	if summary:
		model.summary()

	plot_model(model, to_file='model.png', show_shapes=True)
	return model, 'nvidia'

model, name = nvidia(LR=1e-4, inputshape=INPUT_SHAPE, comp=True, summary=False)

save_path = '/Users/mohammedamarnah/Desktop/SDCProject/save/'
model_path = save_path+'model-'+name+str(datetime.datetime.now()).replace('/', '-').replace('.', '-')+'.h5'
checkpoint = [ModelCheckpoint(model_path,
                             monitor='val_loss',
                             verbose=0,
                             save_best_only=True,
                             mode='auto')]


print("Training model: ", name)
print("Epochs: ", EPOCHS)

history = model.fit_generator(train, nb_epoch=EPOCHS,
                            samples_per_epoch=SAMPLES_PER_EPOCH, nb_val_samples=len(X_valid),
							max_q_size=1, verbose=1, validation_data=valid, callbacks=checkpoint)

####################################################################

utils.h5_to_json(model_path)
