from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape
from keras.layers.convolutional import Conv2D, ZeroPadding2D, UpSampling2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adam, Adadelta

def model_1():
	input1 = Input(shape=(28,28,1))

	#conv1
	block1 = Conv2D(16,(3,3), padding='valid', activation='relu')(input1)
	block1 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(block1)

	#conv2
	block2 = Conv2D(8,(2,2), padding='valid', activation='relu')(block1)
	block2 = MaxPooling2D(pool_size=(2,2), strides=(2,2))(block2)

	#code
	code = Flatten()(block2)
	code1 = Dense(16, activation='relu')(code)
	code2 = Dense(8, activation='relu')(code1)
	#code3 = Dense(2, activation='softmax')(code2)


	#decode
	#decode2 = Dense(8, activation='relu')(code3)
	decode1 = Dense(16, activation='relu')(code2)
	decode = Dense(6*6*8)(decode1)

	#conv3
	block3 = Reshape((6,6,8))(decode)
	block3 = UpSampling2D(size=(2,2))(block3)
	block3 = ZeroPadding2D(padding=(1,1))(block3)
	block3 = Conv2D(16,(2,2), padding='valid', activation='relu')(block3)

	#conv4
	block4 = UpSampling2D(size=(2,2))(block3)
	block4 = ZeroPadding2D(padding=(2,2))(block4)
	block4 = Conv2D(1,(3,3), padding='valid', activation='relu')(block4)


	autoencoder = Model(inputs=input1, outputs=block4)
	encoder = Model(inputs=input1, outputs=code2)

	opt = Adam(lr=1e-3)
	#opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
	autoencoder.compile(loss='mean_squared_error', optimizer=opt)
	autoencoder.summary()
	encoder.summary()
	return (autoencoder,encoder)


def model_2():
	input1 = Input(shape=(784,))

	#code
	fc1 = Dense(512, activation='relu')(input1) #512
	fc1 = Dense(128, activation='relu')(fc1) #128
	code = Dense(32, activation='relu')(fc1) #32


	#decode
	fc2 = Dense(128, activation='relu')(code) #128
	fc2 = Dense(512, activation='relu')(fc2) #512
	predict = Dense(784, activation='relu')(fc2)

	autoencoder = Model(inputs=input1, outputs=predict)
	encoder = Model(inputs=input1, outputs=code)

	opt = Adam(lr=1e-3)
	#opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
	autoencoder.compile(loss='mean_squared_error', optimizer=opt)
	autoencoder.summary()
	encoder.summary()
	return (autoencoder,encoder)


def model_3():
	input1 = Input(shape=(784,))

	#code
	fc1 = Dense(64, activation='relu')(input1) #512
	fc1 = Dense(32, activation='relu')(fc1) #128
	code = Dense(16, activation='relu')(fc1) #32


	#decode
	fc2 = Dense(32, activation='relu')(code) #128
	fc2 = Dense(64, activation='relu')(fc2) #512
	predict = Dense(784, activation='sigmoid')(fc2)

	autoencoder = Model(inputs=input1, outputs=predict)
	encoder = Model(inputs=input1, outputs=code)

	opt = Adam(lr=1e-3)
	#opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
	autoencoder.compile(loss='mean_squared_error', optimizer=opt)
	autoencoder.summary()
	encoder.summary()
	return (autoencoder,encoder)