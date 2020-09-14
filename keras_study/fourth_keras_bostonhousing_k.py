import numpy as np
import matplotlib.pyplot as plt
from keras import layers, models
from keras.datasets import boston_housing
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#train_data.shape -> (404, 13)
#test_data.shape -> (102, 13)
(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

#数据标准化，减去平均值再除以标准差(测试数据也用训练数据的标准差)
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

#模型定义
def build_model():
	network = models.Sequential()
	network.add(layers.Dense(64, activation='relu', input_shape=(13, )))
	network.add(layers.Dense(64, activation='relu'))
	network.add(layers.Dense(1))	#最后输出预测房价，恒等函数
	#损失函数用mes(均方误差), 监控指标为mae(平均绝对误差, 返回误差绝对值)
	network.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
	return network

#利用K折验证输入的数据
k = 4	#将数据分为4个相同的折，每个折的第i-1个分区作为验证集
num_val = len(train_data) // k	#每个分区大小(一定要整除)
num_epochs = 500
mae_list = []

for i in range(k):
	print("当前第" + str(i) + "折：")
	#验证集
	val_x = train_data[i * num_val: (i+1) * num_val]
	val_t = train_labels[i * num_val: (i+1) * num_val]
	#训练集(注意训练集是验证集剩下的，所以要用concatenate在第一维度连接)
	partial_x_train = np.concatenate([train_data[:i * num_val], train_data[(i+1) * num_val:]], axis=0)
	partial_t_train = np.concatenate([train_labels[:i * num_val], train_labels[(i+1) * num_val:]], axis=0)
	
	network = build_model()
	#verbose：静默模式, 详情见https://blog.csdn.net/WILDCHAP_/article/details/107618130
	History = network.fit(partial_x_train, partial_t_train, 
							validation_data=(val_x, val_t), 
							epochs=num_epochs, batch_size=1, verbose=0)
	history_dict = History.history
	#print(history_dict.keys()) ->mae, val_mae, loss, val_loss
	#将验证集的平均绝对误差加入数组
	mae = history_dict['val_mae']
	mae_list.append(mae)
	#print(len(mae))

#求出每一折的平均绝对误差平均值(每一折都经过留num_epochs次)
average_mae_list = []
for i in range(num_epochs):
	for x in mae_list:
		average_mae_list.append(np.mean(x[i]))

#绘制图像
x = range(1, len(average_mae_list)+1)
plt.plot(x, average_mae_list)
plt.xlabel('Epochs')
plt.ylabel('mean_abs_error')
plt.title('mae data')
plt.show()
