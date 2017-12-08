#!/usr/bin/env python
from __future__ import print_function
from build_model import *
import os
import pickle
from keras.utils import np_utils
from keras.models import load_model
import argparse
import numpy as np


PATIENCE = 10

def main():

	parser = argparse.ArgumentParser(prog='train.py')
	parser.add_argument('batch', type=int, default=512)
	parser.add_argument('epoch', type=int, default=20)
	parser.add_argument('pretrain', type=int, default=0)
	parser.add_argument('save_every', type=int, default=1)
	parser.add_argument('model_name', type=str, default='model-40.h5')
	args = parser.parse_args()

	path = ''
	f = open(os.path.join(path,'X_train.pkl'),'rb')
	X_train = pickle.load(f)
	#X_train = X_train.reshape((180000,158,1))
	f.close()
	f = open(os.path.join(path,'Y_train.pkl'),'rb')
	Y_train = pickle.load(f)
	Y_train = np_utils.to_categorical(Y_train,2)
	#Y_train = np.array([1-(1-i)*2 for i in Y_train])
	f.close()
	f = open(os.path.join(path,'X_valid.pkl'),'rb')
	X_valid = pickle.load(f)
	#X_valid = X_valid.reshape((20000,158,1))
	f.close()
	f = open(os.path.join(path,'Y_valid.pkl'),'rb')
	Y_valid = pickle.load(f)
	Y_valid = np_utils.to_categorical(Y_valid,2)
	#Y_valid = np.array([1-(1-i)*2 for i in Y_valid])
	f.close()


	train(args.batch, args.epoch, args.pretrain, args.save_every, X_train, Y_train, X_valid, Y_valid, args.model_name)

def train(batch_size, num_epoch, pretrain, save_every, X_train, Y_train, X_valid, Y_valid, model_name = None):
	
	if pretrain == 0:
		model = model_4()  ### model kind
	else:
		model = load_model(model_name)
	best_metrics = 0.0
	early_stop_counter = 0

	for e in range(num_epoch):
		print('############################')
		print('Model',e+1)
		print('############################')	

		model.fit(X_train,Y_train,batch_size=batch_size,epochs=1,verbose=2)

		loss_and_metrics = model.evaluate(X_valid,Y_valid,batch_size,verbose=0)
		print ('\nloss & metrics:')
		print (loss_and_metrics)


		if loss_and_metrics[1] >= best_metrics:
			best_metrics = loss_and_metrics[1]
			print ("save best score!!"+str(loss_and_metrics[1]))
			early_stop_counter = 0
		else:
			early_stop_counter += 1

		if (e+1) % save_every == 0:
			model.save('model/model-%d.h5'%(e+1))
			print ("save model %d!!"%(e+1))

		if early_stop_counter >= PATIENCE:
			print ('Stop by early stopping')
			print ('Best score: '+str(best_metrics))
			break


def write_data(path, data):
	f = open(path,'wb')
	pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)
	f.close()
	print("%s written!!"%(path))
	print("=============================")
if __name__ == '__main__':
	main()