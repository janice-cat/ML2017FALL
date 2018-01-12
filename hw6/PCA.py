from skimage import io
import sys
import os
import numpy as np

def main():
	X = readfile()
	Y, avg_X = avg_face(X, True)
	'''
	# plot avg face (Q1)
	plot_face(avg_X, 'avg_face.jpg')
	'''

	# plot 4 eigenfaces (Q2)
	s,V = eigen_factorization(Y)
	'''
	plot_eigenface(V,0,'e0.jpg')
	plot_eigenface(V,1,'e1.jpg')
	plot_eigenface(V,2,'e2.jpg')
	plot_eigenface(V,3,'e3.jpg')
	'''


	#reconstruct with eigenfaces (Q3)
	img = io.imread(sys.argv[2])
	img = img.flatten()
	img = img - avg_X
	rec_with_evector(img,V,avg_X,'reconstruction.jpg')

	'''
	#eigenvalues retio (Q4)
	print('eval = ',s[0],s[1],s[2],s[3])
	print('sum of all eval = ',np.sum(s))
	print('ratio e0 = ',s[0]/np.sum(s))
	print('ratio e1 = ',s[1]/np.sum(s))
	print('ratio e2 = ',s[2]/np.sum(s))
	print('ratio e3 = ',s[3]/np.sum(s))
	'''


def readfile():
	X = []
	for img_name in os.listdir(sys.argv[1]):
		img = io.imread(os.path.join(sys.argv[1],img_name))
		img = img.flatten()
		X.append(img)
	X = np.array(X)
	#print (X.shape)
	return(X)


def avg_face(X, flag=True):
	avg_X = np.mean(X, axis=0)
	if (flag == True):
		X = X-avg_X
		return(X, avg_X)
	else:
		return(avg_X)


def eigen_factorization(X):
	U, s, V =  np.linalg.svd(X,full_matrices=False)
	#print('s = ', s.shape,' ,V = ', V.shape)
	return(s,V)


def rec_with_evector(x,V,avg_X,pic_name):
	y = np.zeros((1080000,))
	for i in range(4):
		w = np.dot(x,V[i])
		y += w*V[i]	
	#print(y)
	y = (y+avg_X)
	y = evector_to_eface(y)
	plot_face(y,pic_name)


def mean_shift(x,avg_X):
	x = x+avg_X
	x = x.astype(np.uint8)
	return(x)
	

def evector_to_eface(x):
	x -= np.min(x)
	x /= np.max(x)
	x = (x*255).astype(np.uint8)
	return(x)


def plot_face(x, pic_name):
	x = x.astype(np.uint8)
	x = x.reshape((600,600,3))
	io.imshow(x)
	io.imsave(pic_name, x)


def plot_eigenface(V,i, pic_name):
	x = np.copy(V[i])
	#print(V[i])
	x = evector_to_eface(x)
	plot_face(x, pic_name)
	#print(x)
	#print(diff(x,V[i]))


def diff(x,y):
	return(np.sum(np.abs(x-y)))





if __name__ == '__main__':
	main()
