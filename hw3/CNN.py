# This code is using tensorflow backend
#!/usr/bin/env python
from build_model import *
import os
import pickle
from keras.utils import np_utils
import numpy as np
import pandas as pd
from random import shuffle
from math import floor
import argparse

os.system('echo $CUDA_VISIBLE_DEVICES')
PATIENCE = 100 # The parameter for early stopping

def main():
	parser = argparse.ArgumentParser(prog='CNN.py')###
	parser.add_argument('batch', type=int, default=64)
	parser.add_argument('epoch', type=int, default=1)
	parser.add_argument('pretrain', type=int, default=0)
	parser.add_argument('save_every', type=int, default=1)
	parser.add_argument('model_name', type=str, default='model/model-1')
	parser.add_argument('train_path', type=str, default='./train.csv')
	args = parser.parse_args()
	print (args.pretrain)

	#Read file
	def _shuffle(X, Y):
		randomize = np.arange(len(X))
		np.random.shuffle(randomize)
		return (X[randomize], Y[randomize])

	def split_valid_set(X_all, Y_all, percentage):
		all_data_size = len(X_all)
		valid_data_size = int(floor(all_data_size * percentage))

		X_all, Y_all = _shuffle(X_all, Y_all)

		X_train, Y_train = X_all[0:valid_data_size], Y_all[0:valid_data_size]
		X_valid, Y_valid = X_all[valid_data_size:], Y_all[valid_data_size:]

		return (X_train, Y_train, X_valid, Y_valid)

	##Read file
	N_train = 28709
	Raw = pd.read_csv(args.train_path, sep=',')
	pixels=[]
	labels=[]
	for row in Raw.itertuples():
		labels.append(row.label)
		row_x = row.feature.split()
		pixels.append([int(x)/255. for x in row_x])
	pixels = np.array(pixels)
	pixels = np.reshape(pixels,(N_train,48,48,1))
	labels = np.array(labels)
	labels = np_utils.to_categorical(labels,7)
	train_pixels, train_labels, valid_pixels, valid_labels = split_valid_set(pixels, labels, 0.9)


	#start training
	train(args.batch, args.epoch, args.pretrain, args.save_every, train_pixels, train_labels, valid_pixels, valid_labels, args.model_name)

def train(batch_size, num_epoch, pretrain, save_every, train_pixels, train_labels, valid_pixels, valid_labels, model_name = None):

	if pretrain == 0:
		print("in\n")
		model = model_8()
	else:
		model = load_model(model_name)

	best_metrics = 0.0
	early_stop_counter = 0

	for e in range(num_epoch):
		model.fit(train_pixels, train_labels, batch_size=batch_size, epochs=1)

		loss_and_metrics = model.evaluate(valid_pixels, valid_labels, batch_size)
		print ('\nloss & metrics:')
		print (loss_and_metrics)

		#early stop is a mechanism to prevent your model from overfitting
		
		if loss_and_metrics[1] >= best_metrics:
			best_metrics = loss_and_metrics[1]
			print ("save best score!! "+str(loss_and_metrics[1]))
			early_stop_counter = 0
		else:
			early_stop_counter += 1
		
		if (e+1) % save_every == 0:
			model.save('model/model-%d.h5' %(e+1))
			print ('Saved model %d!' %(e+1))
		
		if early_stop_counter >= PATIENCE:
			print ('Stop by early stopping')
			print ('Best score: '+str(best_metrics))
			break


if __name__ == '__main__':
	main()
