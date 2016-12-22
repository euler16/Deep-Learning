import numpy as np


class NeuralNet:

	def __init__(self,architect):
		'''architect defines the architecture of the Neural Network'''

		'''number of elements in architect gives the number of layers
		the first and last element denote the input and output layer

		The value of each element gives the number of neurons in that 
		layer

		number of neurons in input = number of features
		'''

		self.architect = architect
		self.num_layer = len(architect)
		self.weights = [np.random.randn(early,new) for early,new in zip(size[:-1],size[1:])]
		self.biases  = [np.random.randn(neurons) for neurons in size[1:]]


	def activation(tensor,activ = 'relu'):

		if activ == 'relu':
			return tensor*(tensor>0)

		if activ == 'tanh':
			return np.tanh(tensor)

		if activ == 'sigmoid':
			return 1.0/(1.0 + np.exp(-tensor))


	def sigmoid_prime(tensor):

		tmp = activation(tensor,activ = 'sigmoid')
		return tmp*(1-tmp)


	def backprop():
		pass

	def cost():
		pass


	def train(self,x_train,y_train,batch_size = 1,epochs = 100):
		'''Makes batches, applies feedforward and backprop'''
		'''ASSUMPTIONS:- 
		x_train : num_samples X num_features
		y_train : num_samples X num_output_features'''

		num_samples = len(y_train)
		num_features = self.architect[0]
		num_output_features = self.architect[-1] 

		# making batches
		batch_data = []
		X = 0
		TARGET = 1

		if batch_size == 1:
			batch_data = [ ( x.reshape((1,num_features)) , y.reshape((1,num_output_features)) ) for x,y in zip(x_train,y_train) ]
		
		else:
			batch_data = [ (x_train[k:k+batch_size],y_train[k:k+batch_size])

			               for k in range(0,num_samples,batch_size) ]

		# training starts
		log = []
		for epoch in range(epochs):

			for x,y in batch_data:
				
				# FEEDFORWARD
				prediction = x			
				for weights,biases in zip(self.weights,self.biases):

					prediction = np.dot(prediction,weights)
					for row in prediction:
						row += biases

					prediction = activation(prediction)



		
