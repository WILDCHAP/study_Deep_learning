import numpy as np
from collections import OrderedDict
from layers import *

class TwoLayersNet():
	def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.1):
		self.params = {}
		self.params['W1'] = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
		self.params['b1'] = np.zeros(hidden_size)
		self.params['W2'] = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
		self.params['b2'] = np.zeros(output_size)
		
		self.layers = OrderedDict()
		self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
		self.layers['Relu1'] = Relu()
		self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
		self.lastlayer = SoftmaxWithLoss()
	#forward
	def pridict(self, x):
		for layer in self.layers.values():
			x = layer.forward(x)
		return x
	#loss
	def loss(self, x, t):
		y = self.pridict(x)
		return self.lastlayer.forward(y, t)
	#accuracy
	def accuracy(self, x, t):
		y = self.pridict(x)
		y = np.argmax(y, axis=1)
		t = np.argmax(t, axis=1)
		batch_size = y.shape[0]
		return np.sum(y==t) / batch_size
	#grad
	def gradient(self, x, t):
		grads = {}
		#forward
		self.loss(x, t)
		#backward
		dout = 1
		dout = self.lastlayer.backward(dout)
		
		layers = list(self.layers.values())
		layers.reverse()
		for layer in layers:
			dout = layer.backward(dout)
		
		grads['W1'] = self.layers['Affine1'].dw
		grads['b1'] = self.layers['Affine1'].db
		grads['W2'] = self.layers['Affine2'].dw
		grads['b2'] = self.layers['Affine2'].db
		return grads

