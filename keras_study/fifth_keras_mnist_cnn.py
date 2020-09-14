import numpy as np
import matplotlib.pyplot as plt
from keras import layers, models
from keras.datasets import mnist
from keras.utils import to_categorical

#导入数据
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()

#将数据转换为卷积网络形式(1个通道)
train_data = train_data.reshape((-1, 28, 28, 1))
test_data = test_data.reshape((-1, 28, 28, 1))
#one-hot表示
train_data = train_data.astype('float32') / 255
test_data = test_data.astype('float32') / 255

#将标签标量化(例如第一个数字是3, 那么就转换成0 0 0 1 0 0 0 0 0 0)
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#定义模型(输入--32卷积--2*2 Max池化--64卷积--2*2 Max池化--64卷积--展开--64全连接--输出)
model = models.Sequential()
#卷积核大小3*3*1, FN=32
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
#卷积核大小3*3*1, FN=64
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#展开并连接池化层
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

#编译模型
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

History = model.fit(train_data, train_labels, epochs=5, batch_size=64)

history_dict = History.history
acc_train_list = []
acc_train_list = history_dict['acc']
x = range(1, len(acc_train_list)+1)
plt.plot(x, acc_train_list)
plt.xlabel('epochs')
plt.ylabel('acc')
plt.title('mnist acc')
plt.legend()
plt.show()
