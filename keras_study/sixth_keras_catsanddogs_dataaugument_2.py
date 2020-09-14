import numpy as np
import matplotlib.pyplot as plt
from keras import layers, models, optimizers
from keras.preprocessing import image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#各种数据目录
train_dir = '/home/wildchap/cats_and_dogs_small/train'
test_dir = '/home/wildchap/cats_and_dogs_small/test'
validation_dir = '/home/wildchap/cats_and_dogs_small/validation'

#利用ImageDataGenerator来设置数据增强
train_datagen = image.ImageDataGenerator(
				rescale=1.0/255,
				rotation_range=20,
				width_shift_range=0.2,
				height_shift_range=0.2,
				shear_range=0.2,
				zoom_range=0.2,
				horizontal_flip=True)
#测试数据不使用数据增强
test_datagen = image.ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
					train_dir,
					target_size=(150, 150),
					batch_size=20,
					class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
					validation_dir,
					target_size=(150, 150),
					batch_size=20,
					class_mode='binary')

#定义模型
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

History = model.fit_generator(
			train_generator,
			steps_per_epoch=100,
			epochs=30,
			validation_data=validation_generator,
			validation_steps=50)
			
model.save('cats_and_dogs_small_2.h5')

#绘制其在训练数据和验证数据上的精度
acc = History.history['acc']
val_acc = History.history['val_acc']
loss = History.history['loss']
val_loss = History.history['val_loss']

x = range(1, len(acc)+1)

plt.plot(x, acc, 'bo', label='train accuracy')
plt.plot(x, val_acc, 'b', label='val accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title('accuracy')
plt.legend()
plt.figure()
plt.plot(x, loss, 'bo', label='train loss')
plt.plot(x, val_loss, 'b', label='val loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('loss')
plt.legend()
plt.show()
