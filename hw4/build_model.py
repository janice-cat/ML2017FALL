#!/usr/bin/env python
# -- coding: utf-8 --
from keras.models import Sequential, Model, load_model
from keras.layers import Input, LSTM, Dense, Dropout, Activation, concatenate
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam, SGD
from keras import regularizers
import numpy as np


def model_1():
	model = Sequential()
	model.add(LSTM(512, input_length=158, input_dim=1))
	model.add(Dense(64, activation='relu'))
	model.add(Dense(128, activation='relu'))
	model.add(Dense(2, activation='softmax'))

	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	model.summary()
	return model

def model_2():
	dropout_rate = 0.5
	reg = 0.001
	input_shape = Input(shape=(158,))

	#Embedding
	block1 = Embedding(input_dim=25000, output_dim=128, trainable=True)(input_shape)

	#LSTM
	block1 = LSTM(512, dropout=dropout_rate)(block1)

	#Dense
	block1 = Dense(256, activation='relu')(block1)
	block1 = Dropout(dropout_rate)(block1)

	predict = Dense(2, activation='softmax')(block1)
	
	model = Model(inputs=input_shape, outputs=predict)

	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	model.summary()
	return model

def model_3():  ## Embedding
	dropout_rate = 0.5
	reg = 0.001
	input_shape = Input(shape=(158,))

	#Embedding
	block1 = Embedding(input_dim=25000, output_dim=128, trainable=True)(input_shape)

	#LSTM
	block1 = LSTM(64, return_sequences=True, dropout=dropout_rate)(block1)
	block1 = LSTM(512, dropout=dropout_rate)(block1)

	#Dense
	block1 = Dense(256, activation='relu')(block1)
	block1 = Dropout(dropout_rate)(block1)

	predict = Dense(2, activation='softmax')(block1)
	
	model = Model(inputs=input_shape, outputs=predict)

	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	model.summary()
	return model

def model_4():  ## Embedding, Dropout
	input_shape = Input(shape=(158,))

	#Embedding
	block1 = Embedding(input_dim=25000, output_dim=128, trainable=True)(input_shape)

	#LSTM
	block1 = LSTM(512 , dropout=0.5)(block1)

	#Dense
	block1 = Dense(64, activation='relu')(block1)
	block1 = Dropout(0.5)(block1)
	block1 = Dense(128, activation='relu')(block1)
	block1 = Dropout(0.5)(block1)

	predict = Dense(2, activation='softmax')(block1)
	
	model = Model(inputs=input_shape, outputs=predict)

	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	model.summary()
	return model

def model_5():  ## hinge loss
	dropout_rate = 0.5
	reg = 0.001
	input_shape = Input(shape=(158,))

	#Embedding
	block1 = Embedding(input_dim=25000, output_dim=128, trainable=True)(input_shape)

	#LSTM
	block1 = LSTM(128, dropout=dropout_rate)(block1)

	#Dense
	block1 = Dense(64, activation='relu')(block1)
	block1 = Dropout(dropout_rate)(block1)
	block1 = Dense(128, activation='relu')(block1)
	block1 = Dropout(dropout_rate)(block1)

	predict = Dense(1, activation='linear', W_regularizer=regularizers.l2(0.01))(block1)
	
	model = Model(inputs=input_shape, outputs=predict)

	model.compile(loss='hinge',optimizer='adam',metrics=['accuracy'])
	model.summary()
	return model

def model_6():  ## Embedding, Dropout

	input_shape = Input(shape=(5,))
	block1 = Dense(5, activation='relu')(input_shape)
	predict = Dense(2, activation='relu')(block1)

	model = Model(inputs=input_shape, outputs=predict)
	opt = SGD(lr=5e-2, momentum=0.9)
	model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

	model.summary()
	return model


