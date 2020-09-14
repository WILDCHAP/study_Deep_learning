import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from three_layers_net import ThreeLayersNet
from alter_grad import *

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

#super_data
item_size = 10000
train_size = x_train.shape[0]
batch_size = 100
epoch = train_size / batch_size
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

opt = Momentum()

network = ThreeLayersNet(input_size=784, hidden1_size=30, hidden2_size=50, output_size=10)

for i in range(item_size):
	batch_mask = np.random.choice(train_size, batch_size)
	x_batch = x_train[batch_mask]
	t_batch = t_train[batch_mask]
	grads = network.gradient(x_batch, t_batch)
	#updata_params
	opt.update(network.params, grads)
		
	train_loss = network.loss(x_batch, t_batch)
	train_loss_list.append(train_loss)
	
	if i%epoch==0:
		train_acc = network.accuracy(x_train, t_train)
		test_acc = network.accuracy(x_test, t_test)
		train_acc_list.append(train_acc)
		test_acc_list.append(test_acc)
		print("train_acc:", train_acc, "\ttest_acc:", test_acc)
x = np.arange(len(train_loss_list))
plt.plot(x, train_loss_list, label='loss')
plt.xlabel("train_times")
plt.ylabel("loss")
plt.show()
