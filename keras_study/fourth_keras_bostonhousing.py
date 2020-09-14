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
network = models.Sequential()
network.add(layers.Dense(64, activation='relu', input_shape=(13, )))
network.add(layers.Dense(64, activation='relu'))
network.add(layers.Dense(1))	#最后输出预测房价，恒等函数
#损失函数用mes(均方误差), 监控指标为mae(平均绝对误差, 返回误差绝对值)
network.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])

#由于数据量少, 不引入验证集
History = network.fit(train_data, train_labels, epochs=20, batch_size=80)

#用训练好的模型衡量测试数据精确度
results = network.evaluate(test_data, test_labels)
print(results)

#用训练好的网络预测结果
print(network.predict(test_data))

#绘制图像
history_dict = History.history
print(history_dict.keys())
acc_list = history_dict['mae']

x = range(1, len(acc_list) + 1)

plt.plot(x, acc_list, linestyle=':', label='Training_mae')
plt.title('mae data')
plt.xlabel('Epoches')
plt.ylabel('mean abs error')
plt.legend()
plt.show()
