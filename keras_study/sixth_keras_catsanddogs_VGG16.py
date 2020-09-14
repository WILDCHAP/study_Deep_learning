import os
import numpy as np
import matplotlib.pyplot as plt
from keras import layers, models, optimizers
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator

#将VGG16卷积基实例化
conv_base = VGG16(weights='imagenet',
					include_top=False,
					input_shape=(150, 150, 3))
					
base_dir = '/home/wildchap/cats_and_dogs_small'
train_dir = '/home/wildchap/cats_and_dogs_small/train'
test_dir = '/home/wildchap/cats_and_dogs_small/test'
validation_dir = '/home/wildchap/cats_and_dogs_small/validation'

#不使用数据增强
datagen = ImageDataGenerator(rescale=1.0/255)
batch_size = 20

#将图像和标签转换为VGG16所接受的numpy数组
def extract_features(dir, sample_count):
	features = np.zeros(shape=(sample_count, 4, 4, 512))	#提取特征形状为每图4*4*512
	labels = np.zeros(shape=(sample_count))
	#根据传入目录进行分类标签
	generator = datagen.flow_from_directory(
				dir,
				target_size=(150, 150),
				batch_size=batch_size,
				class_mode='binary')
	i = 0
	for inputs_batch, labels_batch in generator:
		#利用conv_base模型的predict方法来从图像中提取特征
		features_batch = conv_base.predict(inputs_batch)
		features[i * batch_size : (i+1) * batch_size] = features_batch
		labels[i * batch_size : (i+1) * batch_size] = labels_batch
		i += 1
		if i*batch_size >= sample_count:
			break
	
	return features, labels
	
#转换
train_features, train_labels = extract_features(train_dir, 2000)
test_features, test_labels = extract_features(test_dir, 1000)
validation_features, validation_labels = extract_features(validation_dir, 1000)

#由于要和Dense层连接，所以我们要展开
train_features = np.reshape(train_features, (2000, 4*4*512))
test_features = np.reshape(test_features, (1000, 4*4*512))
validation_features = np.reshape(validation_features, (1000, 4*4*512))

#使用Dropout正则化
model = models.Sequential()
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=1e-5),
				loss='binary_crossentropy',
				metrics=['acc'])
				
History = model.fit(train_features, train_labels, epochs=30, batch_size=20, validation_data=(validation_features, validation_labels))

#绘制
acc = History.history['acc']
val_acc = History.history['val_acc']
loss = History.history['loss']
val_loss = History.history['val_loss']

x = range(1, len(acc)+1)

plt.plot(x, acc, 'bo', label='Train acc')
plt.plot(x, val_acc, 'b', label='Validation acc')
plt.xlabel('epochs')
plt.ylabel('accutary')
plt.title('accuracy')
plt.legend()

plt.figure()

plt.plot(x, loss, 'bo', label='Train loss')
plt.plot(x, val_loss, 'b', label='Validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('loss')
plt.legend()

plt.show()

