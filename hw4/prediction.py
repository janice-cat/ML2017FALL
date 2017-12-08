import os
import pickle
from keras.models import load_model
import argparse
import numpy as np
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

def main():
	parser = argparse.ArgumentParser(prog='prediction.py')
	parser.add_argument('model', type=str, default='model/model-1.h5')
	parser.add_argument('testing_data',type=str, default='testing_data.txt')
	parser.add_argument('predict_file',type=str, default='Submission.csv')
	args = parser.parse_args()

	f = open('tokenizer.pkl','rb')
	tokenizer = pickle.load(f)
	f.close()

	f = open(args.testing_data,'r').readlines()
	f.pop(0)
	X_test_text = []
	for i,line in enumerate(f):
		line = line.strip('%d,'%(i))
		X_test_text.append(line)

	X_test_text = np.array(X_test_text)
	X_test = to_RNN(tokenizer, X_test_text, 158)

	prediction(args.model, X_test, args.predict_file)

def prediction(model_name, X_test, file):
	model = load_model(model_name)
	Y_test_onehot = model.predict(X_test)

	f = open(file,'w')
	f.write('id,label\n')
	Y_test = np.argmax(Y_test_onehot, axis=1)
	for i,Y in enumerate(Y_test):
		f.write('%d,%d\n'%(i,Y))
	f.close()

def to_RNN(tokenizer, X_text, max_len):
	X_RNN = tokenizer.texts_to_sequences(X_text)
	X_RNN = pad_sequences(X_RNN, maxlen=max_len)
	return(X_RNN)

if __name__ == '__main__':
	main()