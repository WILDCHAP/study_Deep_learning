from keras.datasets import mnist
from keras import models, layers
from keras.utils import to_categorical
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#导入数据
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#构建结构
network = models.Sequential()
network.add(layers.Dense(512, activation='relu'))
network.add(layers.Dense(10, activation='softmax'))

#编译
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#规范数据
train_images = train_images.reshape(60000, 28*28)
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape(10000, 28*28)
test_images = test_images.astype('float') / 255

#准备标签
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#训练
network.fit(train_images, train_labels, epochs=5, batch_size=128)
