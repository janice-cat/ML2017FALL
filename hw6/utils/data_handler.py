import numpy as np
from random import shuffle
from math import floor
import pickle
import os

def _shuffle(X,Y,Z=None):
	randomize = np.arange(len(X))
	np.random.shuffle(randomize)
	if Z is None:
		return(X[randomize],Y[randomize])
	elif Z is not None:
		return(X[randomize],Y[randomize],Z[randomize])

def split_set(X_all,Y_all,percentage):
	all_data_size = len(X_all)
	train_data_size = int(floor(all_data_size*percentage))

	X_all, Y_all = _shuffle(X_all, Y_all)

	X_train, Y_train = X_all[0:train_data_size],Y_all[0:train_data_size]
	X_valid, Y_valid = X_all[train_data_size:],Y_all[train_data_size:]

	return(X_train, Y_train, X_valid, Y_valid)

def write_data(path, data, mode):

	f = open(path,mode)
	pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)
	f.close()
	print("%s written!!"%(path))
	print("=============================")

def write_data_3to2(path, data, mode):

	f = open(path,mode)
	pickle.dump(data,f,2)
	f.close()
	print("%s written!!"%(path))
	print("=============================")

def load_data(path, mode, encoding='utf-8'):
	f = open(path,mode)
	data = pickle.load(f, encoding=encoding)
	f.close()
	print("%s read!!"%(path))
	print("=============================")
	return(data)

def pad(Y,maxlen,fix_length):
	N = Y.shape[0]
	Z = np.zeros(shape=(N,maxlen,fix_length))
	for i in range(N):
		Z[i,maxlen-Y[i].shape[0]:,:] = Y[i]
	return(Z)