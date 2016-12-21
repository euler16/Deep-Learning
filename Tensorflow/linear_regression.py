import tensorflow as tf
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

def get_data(filename,frac):
	
	data = pd.read_csv(filename)
	LABEL = 'MEDV'

	data['x0'] = np.array([1 for i in range(len(data.index))])

	test_data = data.sample(frac = frac)
	train_data = data.drop(test_data.index)

	from sklearn import preprocessing as prp

	y_test = np.array(test_data[LABEL])
	y_test = y_test.reshape((len(y_test),1))   # MAKING it 2 dimensional for tensorflow
	x_test = test_data.drop(LABEL,axis = 1).values
	std_scale = prp.StandardScaler().fit(x_test)
	x_test = std_scale.transform(x_test)

	y_train = np.array(train_data[LABEL])
	y_train = y_train.reshape((len(y_train),1))
	x_train = train_data.drop(LABEL,axis = 1).values
	std_scale = prp.StandardScaler().fit(x_train)
	x_train = std_scale.transform(x_train)

	return x_test, y_test, x_train, y_train	


filename = "BostonHousingPrices/Boston_Housing_Prices.csv"

x_test, y_test, x_train, y_train = get_data(filename,frac = 0.3)

from sklearn.linear_model import LinearRegression
regr = LinearRegression(fit_intercept = True)
regr.fit(x_train,y_train)
wsk = regr.coef_


n_samples,n_features = x_test.shape


# HYPERPARAMETER 

BATCH_SIZE = 1
ETA = 0.0003
epochs = 1000

X_TRAIN = tf.placeholder(tf.float64,shape = (None,n_features),name = "train_data")
Y_TRAIN = tf.placeholder(tf.float64,shape = (None,1), name = "train_labels")

W = tf.Variable(np.random.randn(n_features,1),name = "weights")
# B = tf.Variable(np.random.randn(None,1),name = "biases")

prediction = tf.matmul(X_TRAIN,W)
cost = tf.reduce_sum(tf.pow(Y_TRAIN-prediction,2))/2

optimizer = tf.train.GradientDescentOptimizer(ETA).minimize(cost)
init_op = tf.initialize_all_variables()

log = []

with tf.Session() as sess:

	sess.run(init_op)

	for epoch in range(epochs):

		for x,y in zip(x_train,y_train):
			sess.run(optimizer,feed_dict = {X_TRAIN:x.reshape(((x.size/n_features),n_features)),Y_TRAIN:y.reshape((y.size,1))})

		log.append(sess.run(cost,feed_dict = {X_TRAIN:x_train,Y_TRAIN:y_train}))
		# print("Epoch: ",epoch,"    cost: ",log[epoch])

	print("Optimization finished\n")

	weights = sess.run(W)
	print("Weights:  ",weights)
	print("sklearn weights: ",wsk)
	test_prediction = sess.run(tf.matmul(x_test[0].reshape((1,n_features)),weights))
	print("\n\n",test_prediction," ",y_test[0])
	print(regr.predict(x_test[0]))

plt.plot([i for i in range(epochs)],log)
plt.show()