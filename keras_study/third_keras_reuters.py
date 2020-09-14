import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import reuters
from keras import layers, models
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#8982个训练样本和2246个测试样本
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

#编码，方法同处理imdb二分类(f==1:处理数据，f==2:处理标签)
def vectorize_sequences(sequences, f):
	if f == 1:
		dimension = 10000
	else:
		dimension = 46
	results = np.zeros((len(sequences), dimension))
	for i,k in enumerate(sequences):
		results[i, k] = 1
	return results
	
x_train = vectorize_sequences(train_data, 1)
t_train = vectorize_sequences(train_labels, 2)
x_test = vectorize_sequences(test_data, 1)
t_test = vectorize_sequences(test_labels, 2)

#预留出验证集
val_x = x_train[:1000]
partial_x_train = x_train[1000:]
val_t = t_train[:1000]
partial_t_train = t_train[1000:]

#定义模型
network = models.Sequential()
network.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
network.add(layers.Dense(64, activation='relu'))
network.add(layers.Dense(46, activation='softmax'))

#编译模型
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#训练模型
History = network.fit(partial_x_train, partial_t_train, epochs=20, batch_size=512, validation_data=(val_x, val_t))

#用训练好的模型衡量测试数据精确度
results = network.evaluate(x_test, t_test)
print(results)

#用训练好的网络预测结果
predictions = network.predict(x_test)
print("预测形状" + str(predictions[0].shape))
print("每一维度向量和" + str(np.sum(predictions[0])))
print("第一个新闻种类" + str(np.argmax(predictions[0])))


#绘制准确率
history_dict = History.history
acc_list = history_dict['accuracy']
val_acc_list = history_dict['val_accuracy']

x = range(1, len(acc_list)+1)

plt.plot(x, acc_list, linestyle=':', label='train_acc')
plt.plot(x, val_acc_list, linestyle='-', label='val_acc')
plt.title('acc data')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()



