import numpy as np

def softmax(x):
	c = np.max(x, axis=1, keepdims=True)
	f_z = np.exp(x-c)
	f_m = np.sum(f_z, axis=1, keepdims=True)
	return f_z / f_m

def cross_entropy_error(y, t):
	if y.ndim==1:
		y = y.reshape(1, y.shape)
		t = t.reshape(1, t.shape)
	batch_size = y.shape[0]
	return -np.sum(t*np.log(y+1e-9)) / batch_size
