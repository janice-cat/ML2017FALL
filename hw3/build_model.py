#!/usr/bin/env python
# -- coding: utf-8 --
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adam, Adadelta
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import numpy as np

def model_1():
	model = Sequential()
	model.add(Conv2D(25,(5,5),input_shape=(48,48,1)))
	model.add(MaxPooling2D((2,2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(25,(3,3)))
	model.add(MaxPooling2D((2,2)))
	model.add(Dropout(0.25))

	model.add(Flatten())


	model.add(Dense(200, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(7, activation='softmax'))

	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	model.summary()
	return model

def model_2():
	input_img = Input(shape=(48,48,1))

	##conv1
	block1 = Conv2D(64,(5,5), padding='valid', activation='relu')(input_img)
	block1 = ZeroPadding2D(padding=(2,2), data_format='channels_last')(block1)
	block1 = MaxPooling2D(pool_size=(5,5), strides=(2,2))(block1)
	block1 = ZeroPadding2D(padding=(1,1), data_format='channels_last')(block1)

	##conv2
	block2 = Conv2D(64,(3,3), activation='relu')(block1)
	block2 = ZeroPadding2D(padding=(1,1), data_format='channels_last')(block2)

	##conv3
	block3 = Conv2D(64,(3,3), activation='relu')(block2)
	block3 = AveragePooling2D(pool_size=(3,3), strides=(2,2))(block3)
	block3 = ZeroPadding2D(padding=(1,1), data_format='channels_last')(block3)

	#conv4
	block4 = Conv2D(128,(3,3), activation='relu')(block3)
	block4 = ZeroPadding2D(padding=(1,1), data_format='channels_last')(block4)

	#conv5
	block5 = Conv2D(128,(3,3), activation='relu')(block4)
	block5 = ZeroPadding2D(padding=(1,1), data_format='channels_last')(block5)
	block5 = AveragePooling2D(pool_size=(3,3), strides=(2,2))(block5)
	block5 = Flatten()(block5)

	#Dense1
	fc1 = Dense(1024, activation='relu')(block5)
	fc1 = Dropout(0.5)(fc1)

	#Dense2
	fc2 = Dense(1024, activation='relu')(fc1)
	fc2 = Dropout(0.5)(fc2)

	predict = Dense(7)(fc2)
	predict = Activation('softmax')(predict)
	model = Model(inputs=input_img, outputs=predict)

	#opt = Adam(lr=1e-3)
	opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	model.summary()
	return model


def model_3(): ### more dropout (5 conv)
	input_img = Input(shape=(48,48,1))

	##conv1
	block1 = Conv2D(64,(5,5), padding='valid', activation='relu')(input_img)
	block1 = ZeroPadding2D(padding=(2,2), data_format='channels_last')(block1)
	block1 = MaxPooling2D(pool_size=(5,5), strides=(2,2))(block1)
	block1 = ZeroPadding2D(padding=(1,1), data_format='channels_last')(block1)
	#block1 = Dropout(0.5)(block1)

	##conv2
	block2 = Conv2D(64,(3,3), activation='relu')(block1)
	block2 = ZeroPadding2D(padding=(1,1), data_format='channels_last')(block2)
	#block2 = Dropout(0.5)(block2)

	##conv3
	block3 = Conv2D(64,(3,3), activation='relu')(block2)
	block3 = AveragePooling2D(pool_size=(3,3), strides=(2,2))(block3)
	block3 = ZeroPadding2D(padding=(1,1), data_format='channels_last')(block3)
	#block3 = Dropout(0.5)(block3)

	#conv4
	block4 = Conv2D(128,(3,3), activation='relu')(block3)
	block4 = ZeroPadding2D(padding=(1,1), data_format='channels_last')(block4)
	block4 = Dropout(0.5)(block4)

	#conv5
	block5 = Conv2D(128,(3,3), activation='relu')(block4)
	block5 = ZeroPadding2D(padding=(1,1), data_format='channels_last')(block5)
	block5 = AveragePooling2D(pool_size=(3,3), strides=(2,2))(block5)
	block5 = Dropout(0.5)(block5)
	block5 = Flatten()(block5)

	#Dense1
	fc1 = Dense(1024, activation='relu')(block5)
	fc1 = Dropout(0.5)(fc1)

	#Dense2
	fc2 = Dense(1024, activation='relu')(fc1)
	fc2 = Dropout(0.5)(fc2)

	predict = Dense(7)(fc2)
	predict = Activation('softmax')(predict)
	model = Model(inputs=input_img, outputs=predict)

	opt = Adam(lr=1e-3)
	#opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	model.summary()
	return model


def model_4(): ### more layers (6 conv)
	input_img = Input(shape=(48,48,1))

	##conv1
	block1 = Conv2D(32,(3,3), activation='relu')(input_img)
	block1 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(block1)

	##conv2
	block2 = Conv2D(64,(4,4), activation='relu')(block1)
	block2 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(block2)

	##conv3
	block3 = Conv2D(64,(5,5), activation='relu')(block2)
	block3 = Flatten()(block3)

	#Dense1
	fc1 = Dense(256, activation='relu')(block3)

	#Dense2
	fc2 = Dense(512, activation='relu')(fc1)

	#Dense3
	fc3 = Dense(512, activation='relu')(fc2)

	predict = Dense(7)(fc3)
	predict = Activation('softmax')(predict)
	model = Model(inputs=input_img, outputs=predict)

	opt = Adam(lr=1e-3)
	#opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	model.summary()
	return model


def model_5(): ### more dropout (4 conv)
	input_img = Input(shape=(48,48,1))

	##conv1
	block1 = Conv2D(64,(5,5), padding='valid', activation='relu')(input_img)
	block1 = ZeroPadding2D(padding=(2,2), data_format='channels_last')(block1)
	block1 = MaxPooling2D(pool_size=(5,5), strides=(2,2))(block1)
	block1 = ZeroPadding2D(padding=(1,1), data_format='channels_last')(block1)
	#block1 = Dropout(0.5)(block1)

	##conv2
	block2 = Conv2D(64,(3,3), activation='relu')(block1)
	block2 = AveragePooling2D(pool_size=(3,3), strides=(2,2))(block2)
	block2 = ZeroPadding2D(padding=(1,1), data_format='channels_last')(block2)

	#conv3
	block3 = Conv2D(128,(3,3), activation='relu')(block2)
	block3 = ZeroPadding2D(padding=(1,1), data_format='channels_last')(block3)
	block3 = Dropout(0.5)(block3)

	#conv4
	block4 = Conv2D(128,(3,3), activation='relu')(block3)
	block4 = ZeroPadding2D(padding=(1,1), data_format='channels_last')(block4)
	block4 = AveragePooling2D(pool_size=(3,3), strides=(2,2))(block4)
	block4 = Dropout(0.5)(block4)
	block4 = Flatten()(block4)

	#Dense1
	fc1 = Dense(1024, activation='relu')(block4)
	fc1 = Dropout(0.5)(fc1)

	#Dense2
	fc2 = Dense(1024, activation='relu')(fc1)
	fc2 = Dropout(0.5)(fc2)

	predict = Dense(7)(fc2)
	predict = Activation('softmax')(predict)
	model = Model(inputs=input_img, outputs=predict)

	def tsne(P, activations):
		#     d = K.shape(activations)[1]
		d = 2 # TODO: should set this automatically, but the above is very slow for some reason
		n = 128 # TODO: should set this automatically
		v = d - 1.
		eps = K.variable(10e-15) # needs to be at least 10e-8 to get anything after Q /= K.sum(Q)
		sum_act = K.sum(K.square(activations), axis=1)
		Q = K.reshape(sum_act, [-1, 1]) + -2 * K.dot(activations, K.transpose(activations))
		Q = (sum_act + Q) / v
		Q = K.pow(1 + Q, -(v + 1) / 2)
		Q *= K.variable(1 - np.eye(n))
		Q /= K.sum(Q)
		Q = K.maximum(Q, eps)
		C = K.log((P + eps) / (Q + eps))
		C = K.sum(P * C)
		return C

	#opt = Adam(lr=1e-3)
	opt = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	#opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
	model.compile(loss=tsne, optimizer=opt, metrics=['accuracy'])
	model.summary()
	return model


def model_6():
	input_img = Input(shape=(48,48,1))

	##conv1
	block1 = Conv2D(32,(3,3), activation='relu')(input_img)
	block1 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(block1)

	##conv2
	block2 = Conv2D(64,(4,4), activation='relu')(block1)
	block2 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(block2)

	##conv3
	block3 = Conv2D(64,(5,5), activation='relu')(block2)

	##conv4
	block4 = Conv2D(64,(5,5), activation='relu')(block3)
	block4 = Flatten()(block4)

	#Dense1
	fc1 = Dense(256, activation='relu')(block4)
	fc1 = Dropout(0.5)(fc1)

	#Dense2
	fc2 = Dense(512, activation='relu')(fc1)
	fc2 = Dropout(0.5)(fc2)

	predict = Dense(7)(fc2)
	predict = Activation('softmax')(predict)
	model = Model(inputs=input_img, outputs=predict)

	opt = Adam(lr=1e-3)
	#opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	model.summary()
	return model



def model_7(): ### more dropout (4 conv)
	input_img = Input(shape=(48,48,1))

	##conv1
	block1 = Conv2D(128,(5,5), padding='valid', activation='relu')(input_img)
	block1 = ZeroPadding2D(padding=(2,2), data_format='channels_last')(block1)
	block1 = MaxPooling2D(pool_size=(5,5), strides=(2,2))(block1)
	block1 = ZeroPadding2D(padding=(1,1), data_format='channels_last')(block1)
	#block1 = Dropout(0.5)(block1)

	##conv2
	block2 = Conv2D(128,(3,3), activation='relu')(block1)
	block2 = AveragePooling2D(pool_size=(3,3), strides=(2,2))(block2)
	block2 = ZeroPadding2D(padding=(1,1), data_format='channels_last')(block2)

	#conv3
	block3 = Conv2D(256,(3,3), activation='relu')(block2)
	block3 = ZeroPadding2D(padding=(1,1), data_format='channels_last')(block3)
	block3 = AveragePooling2D(pool_size=(3,3), strides=(2,2))(block3)
	block3 = Dropout(0.5)(block3)
	block3 = Flatten()(block3)

	#Dense1
	fc1 = Dense(512, activation='relu')(block3)
	fc1 = Dropout(0.5)(fc1)

	#Dense2
	fc2 = Dense(512, activation='relu')(fc1)
	fc2 = Dropout(0.5)(fc2)
	
	#Dense3
	fc3 = Dense(1024, activation='relu')(fc2)
	fc3 = Dropout(0.5)(fc3)
	
	predict = Dense(7)(fc3)
	predict = Activation('softmax')(predict)
	model = Model(inputs=input_img, outputs=predict)

	opt = Adam(lr=1e-3)
	#opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	model.summary()
	return model


def model_8(): ### more dropout (4 conv) + small
	input_img = Input(shape=(48,48,1))

	##conv1
	block1 = Conv2D(32,(5,5), padding='valid', activation='relu')(input_img)
	block1 = ZeroPadding2D(padding=(2,2), data_format='channels_last')(block1)
	block1 = MaxPooling2D(pool_size=(5,5), strides=(2,2))(block1)
	block1 = ZeroPadding2D(padding=(1,1), data_format='channels_last')(block1)
	#block1 = Dropout(0.5)(block1)

	##conv2
	block2 = Conv2D(64,(3,3), activation='relu')(block1)
	block2 = AveragePooling2D(pool_size=(3,3), strides=(2,2))(block2)
	block2 = ZeroPadding2D(padding=(1,1), data_format='channels_last')(block2)

	#conv3
	block3 = Conv2D(64,(3,3), activation='relu')(block2)
	block3 = ZeroPadding2D(padding=(1,1), data_format='channels_last')(block3)
	block3 = Dropout(0.5)(block3)

	#conv4
	block4 = Conv2D(128,(3,3), activation='relu')(block3)
	block4 = ZeroPadding2D(padding=(1,1), data_format='channels_last')(block4)
	block4 = AveragePooling2D(pool_size=(3,3), strides=(2,2))(block4)
	block4 = Dropout(0.5)(block4)
	block4 = Flatten()(block4)

	#Dense1
	fc1 = Dense(1024, activation='relu')(block4)
	fc1 = Dropout(0.5)(fc1)

	#Dense2
	fc2 = Dense(1024, kernel_regularizer=regularizers.l2(0.001), activation='relu')(fc1)
	fc2 = Dropout(0.5)(fc2)

	predict = Dense(7)(fc2)
	predict = Activation('softmax')(predict)
	model = Model(inputs=input_img, outputs=predict)

	opt = Adam(lr=1e-3)
	#opt = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	#opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	model.summary()
	return model
	
def model_9(): # DNN
	input_img = Input(shape=(48,48,1))

	#Dense1
	fc1 = Flatten()(input_img)
	fc1 = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.001))(fc1)
	#fc1 = Dropout(0.3)(fc1)

	#Dense2
	fc2 = Dense(1024, activation='relu')(fc1)
	fc12 = Dropout(0.5)(fc2)

	#Dense3
	fc3 = Dense(1024, activation='relu')(fc2)
	fc3 = Dropout(0.5)(fc3)

	predict = Dense(7)(fc3)
	predict = Activation('softmax')(predict)
	model = Model(inputs=input_img, outputs=predict)

	opt = Adam(lr=1e-3)
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	model.summary()
	return model