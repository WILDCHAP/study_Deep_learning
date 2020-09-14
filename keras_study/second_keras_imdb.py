import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import imdb
from keras import models, layers
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#加载imdb数据集(仅保留数据集中前10000个最常出现的单词)
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print(len(test_data))
#虚拟机内存不足，减小训练数据大小
train_data = train_data[:10000]
train_labels = train_labels[:10000]
test_data = test_data[:10000]
test_labels = test_labels[:10000]

#函数功能：将上面的整数序列转换成二进制矩阵(进行one-hot编码)
#例如：某条评论为“so bad movie”，原本存储为[1256, 589, 6984]
#现在要将其转换为3*10000的矩阵(3个单词)，矩阵中只有0和1，第一个1的位置在第一行第1256个位置
def vectorize_sequences(sequences, dimension=10000):
	results = np.zeros((len(sequences), dimension))
		#先填充好一个形状为(len(sequences), dimension)的零矩阵
	for i, data in enumerate(sequences):
		results[i, data] = 1
	return results

#将数据向量化
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
#标签向量化
t_train = np.asarray(train_labels).astype('float32')
t_test = np.asarray(test_labels).astype('float32')
#将训练数据留2000个作为验证集
x_val = x_train[:2000]
partial_x_train = x_train[2000:]
t_val = t_train[:2000]
partial_t_train = t_train[2000:]


#定义模型
network = models.Sequential()
network.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
network.add(layers.Dense(16, activation='relu'))
network.add(layers.Dense(1, activation='sigmoid'))

#编译模型
network.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

#训练模型
History = network.fit(partial_x_train, partial_t_train,
						epochs=20, batch_size=512,
						validation_data=(x_val, t_val))

#用训练好的模型衡量测试数据精确度
results = network.evaluate(x_test, t_test)
print(results)

#用训练好的网络预测结果
print(network.predict(x_test))


#绘制图像
history_dict = History.history
loss_list = history_dict['loss']
val_loss_list = history_dict['val_loss']
acc_list = history_dict['accuracy']
val_acc_list = history_dict['val_accuracy']

#x = range(1, len(loss_list) + 1)
x = range(1, len(acc_list) + 1)

plt.plot(x, acc_list, linestyle=':', label='Training_acc')
plt.plot(x, val_acc_list, linestyle='-', label='validation_acc')
plt.title('accuracy data')
plt.xlabel('Epoches')
plt.ylabel('acc')
plt.legend()
plt.show()

