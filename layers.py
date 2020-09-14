import numpy as np
from function import *

#Affine
class Affine():
	def __init__(self, w, b):
		self.W = w
		self.b = b
		self.x = None
		self.dw = None
		self.db = None
	def forward(self, x):
		self.x = x
		return np.dot(x, self.W) + self.b
	def backward(self, dout):
		self.dw = np.dot(self.x.T, dout)
		self.db = np.sum(self.b, axis=0)
		return np.dot(dout, self.W.T)

#Relu
class Relu():
	def __init__(self):
		self.mask = None
	def forward(self, x):
		self.mask = (x<=0)
		out = x.copy()
		out[self.mask] = 0
		return out
	def backward(self, dout):
		dout[self.mask] = 0
		return dout

#SoftmaxWithLoss
class SoftmaxWithLoss():
	def __init__(self):
		self.y = None
		self.t = None
	def forward(self, x, t):
		self.y = softmax(x)
		self.t = t
		return cross_entropy_error(self.y, self.t)
	def backward(self, dout):
		batch_size = self.y.shape[0]
		return (self.y - self.t) / batch_size
