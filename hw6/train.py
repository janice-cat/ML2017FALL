from build_model import *
import os
import pickle
from keras.models import load_model
#from keras.callbacks import LambdaCallback
import numpy as np
import argparse
import time
import sys
import utils.model_utils as MU
from sklearn.cluster import KMeans

def main():
	X_train = np.load('data/image.npy')
	# shape = (140000, 784)
	X_train = X_train/255.
	#X_train = X_train.reshape((140000,28,28,1))

	train(2048, 20, X_train, X_train)



def train(batch_size, num_epoch, X_train, Y_train):
	
	autoencoder,encoder = model_2()
	#autoencoder = load_model('model/autoencoder-40.h5')
	#encoder = load_model('model/encoder-40.h5')

	best_metrics = 0.0
	early_stop_counter = 0

	for e in range(num_epoch):
		print('############################')
		print('Model',e+1)
		print('############################')	
		autoencoder.fit(X_train,Y_train,batch_size=batch_size,epochs=1, validation_split=0.08)

		if e%2 == 0:
			predict('kmeans', encoder, X_train)

		autoencoder.save('model/autoencoder-%d.h5'%(e+1))
		encoder.save('model/encoder-%d.h5'%(e+1))
		print ("save model %d!!"%(e+1))
		

	return(code)


def predict(way, encoder, X_train):
	code = get_codes(encoder, X_train)
	if way == 'kmeans':
		code_K = predict_by_KMeans(code)
		MU.draw(code[:1000],code_K[:1000])
	return(code_K)

def get_codes(encoder, X_train):
	code = encoder.predict(X_train)
	print(code[:100])
	return(code)

def predict_by_KMeans(code):
	kmeans = KMeans(n_clusters=2).fit(code)
	print('Inertia = ', kmeans.inertia_)
	return(kmeans.labels_)

if __name__ == '__main__':
	main()