from keras.models import load_model
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import sys
import utils.model_utils as MU



def main():
	X_train = np.load(sys.argv[2])
	# shape = (140000, 784)
	X_train = X_train/255.
	#X_train = X_train.reshape((140000,28,28,1))
	encoder = load_model(sys.argv[1])
	code = predict('kmeans', encoder, X_train)

	test = load_data()

	make_submission(test, code)





def load_data():
	test = []
	Raw = pd.read_csv(sys.argv[3],header=0)
	for row in Raw.itertuples():
		test.append([row.image1_index, row.image2_index])
	test = np.array(test)
	return(test)


def predict(way, encoder, X_train):
	code = get_codes(encoder, X_train)
	if way == 'kmeans':
		code_K = predict_by_KMeans(code)
		#MU.draw(code[:1000],code_K[:1000])
	return(code_K)

def get_codes(encoder, X_train):
	code = encoder.predict(X_train)
	#print(code[:100])
	return(code)

def predict_by_KMeans(code):
	kmeans = KMeans(n_clusters=2).fit(code)
	#print('Inertia = ', kmeans.inertia_)
	return(kmeans.labels_)

def make_submission(test, code):
	f = open(sys.argv[4],'w')
	f.write('ID,Ans\n')

	for i,t in enumerate(test):
		code_1 = code[t[0]]
		code_2 = code[t[1]]
		if(code_1 == code_2):
			f.write('%d,1\n'%(i))
		else:
			f.write('%d,0\n'%(i))

	f.close()
if __name__ == '__main__':
	main()
