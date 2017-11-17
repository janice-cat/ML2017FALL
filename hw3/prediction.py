import sys
import pickle
from keras.models import load_model
import numpy as np
import pandas as pd
def main():
	#Read file
	N_test = 7178
	Raw = pd.read_csv(sys.argv[1], sep=',')
	test_pixels=[]
	for row in Raw.itertuples():
		row_x = row.feature.split()
		test_pixels.append([int(x)/255. for x in row_x])
	test_pixels = np.array(test_pixels)
	test_pixels = np.reshape(test_pixels,(N_test,48,48,1))

	#start predict
	prediction(test_pixels)

def prediction(test_pixels):
	model = load_model('model-100.h5?dl=1')### Model Kind
	N_test = 7178
	test_labels_onehot = model.predict(test_pixels)

	f = open(sys.argv[2],'w')
	f.write('id,label\n')
	test_labels = np.argmax(test_labels_onehot, axis=1)
	for k in range(N_test):
		f.write('%d,%d\n'%(k,test_labels[k]))
	f.close()

if __name__ == '__main__':
	main()