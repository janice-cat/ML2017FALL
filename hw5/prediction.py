import os
import pickle
from keras.models import load_model
import numpy as np
import sys
import utils.data_handler as DH
from keras import backend as K
import pandas as pd

def main():

	X_test1, X_test2 = load_data()

	X_test1_feature, X_test1_id = sep_data(X_test1)
	X_test2_feature, X_test2_id = sep_data(X_test2)

	#prediction(sys.arg[5], [X_test1_id,X_test2_id,X_test1_feature,X_test2_feature])
	prediction(sys.argv[5], [X_test1_id,X_test2_id])

def sep_data(X):
	N = X.shape[1]
	X_feature = X[:, :N-1]
	X_id = X[:, N-1]
	print(X_feature.shape, X_id.shape)
	return(X_feature, X_id)

def normalization_backward(var, mu, Y_normalized):

	Y_size = Y_normalized.shape
	I = np.ones(shape=Y_size)
	Y = Y_normalized*var + mu*I
	return(Y)

def prediction(model_name, X_test):
	model = load_model(model_name, custom_objects={'RMSE':RMSE})
	Y_test= model.predict(X_test)
	print('prediction completed!!')

	### normalization_backward
	#Y_test = normalization_backward(1.11689766115, 3.58171208604, Y_test)

	f = open(sys.argv[2],'w')
	f.write('TestDataID,Rating\n')

	for i,Y in enumerate(Y_test):
		f.write('%d,%f\n'%(i+1,Y))
	f.close()

def RMSE(y_true, y_pred):
	return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def load_data():
	movie_list = ['Animation',"Children's",'Comedy','Fantasy','Romance','Drama','Crime','Thriller','Adventure','Action','Horror','Sci-Fi','Documentary','War','Musical','Film-Noir','Mystery']

	### train data
	user = []
	user_id = []
	Raw = pd.read_csv(sys.argv[4], sep=':', header=0)

	for row in Raw.itertuples():
		user_id.append(int(row.UserID))

		if (row.Gender == 'F'):
			gender = 0
		elif (row.Gender == 'M'):
			gender = 1

		age = int(row.Age)

		occupation = int(row.Occupation)


		trait_vec = [gender,age,occupation,int(row.UserID)]
		user.append(trait_vec)

	user = np.array(user)


	movie = []
	movie_id = []
	f = open(sys.argv[3], 'r', encoding='latin1').readlines()

	f.pop(0)
	for line in f:
		line = line.strip('\n')
		line = line.split('::')

		movie_id.append(int(line[0]))

		genre = (line[2]).split('|')

		trait_vec = [0 for i in range(17)]

		for i in genre:
			if i in movie_list:
				trait_vec[movie_list.index(i)] += 1

		trait_vec.append(int(line[0]))
		movie.append(trait_vec)

	movie = np.array(movie)


	test_matrix = np.zeros(shape=(6040,3883))
	Raw = pd.read_csv(sys.argv[1], sep=',', header=0)	
	X_test1 = []
	X_test2 = []
	for row in Raw.itertuples():	

		u = int(row.UserID)
		v = int(row.MovieID)

		X_test1.append(user[user_id.index(u)])
		X_test2.append(movie[movie_id.index(v)])

		#test_matrix[user_id.index(u), movie_id.index(v)] = 1

	X_test1 = np.array(X_test1)
	X_test2 = np.array(X_test2)

	
	return(X_test1, X_test2)

if __name__ == '__main__':
	main()