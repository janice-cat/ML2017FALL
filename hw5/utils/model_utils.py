import os
import numpy as np
from keras.models import load_model
from keras.utils.vis_utils import plot_model



def train(model, batch_size, num_epoch, patience, X_train, Y_train, X_valid, Y_valid, model_dir):
	
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

		model.save(os.path.join(model_dir,'model-%d.h5'%(e+1)))
		print ("save model %d!!"%(e+1))

		if early_stop_counter >= patience:
			print ('Stop by early stopping')
			print ('Best score: '+str(best_metrics))
			break

def submission_creator(model_name, X_test):
	model = load_model(model_name)
	Y_test_onehot = model.predict(X_test)
	print('prediction completed!!')

	f = open('Submission.csv','w')
	f.write('id,label\n')
	Y_test = np.argmax(Y_test_onehot, axis=1)
	for i,Y in enumerate(Y_test):
		f.write('%d,%d\n'%(i,Y))
	f.close()

def plot_model_struct(model, model_path):
	classifier = load_model(model)
	classifier.summary()
	plot_model(classifier, to_file=model_path)

