import os
import numpy as np
import matplotlib.pyplot as plt
from keras import layers, models
from keras.preprocessing import image

#利用ImageDataGenerator来设置数据增强
datagen = image.ImageDataGenerator(
			rotation_range=40,		#旋转值
			width_shift_range=0.2,	#水平和垂直方向平移的距离
			height_shift_range=0.2,
			shear_range=0.2,		#随机错切变换
			zoom_range=0.2,			#随机缩放范围
			horizontal_flip=True,	#水平对称翻转
			fill_mode='nearest')

train_cats_dir = '/home/wildchap/cats_and_dogs_small/train/cats'

fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]

#print(fnames)：输出测试数据里的所有文件路径
#print(fnames[3]):输出一个路径

img_path = fnames[3]

img = image.load_img(img_path, target_size=(150, 150))

x = image.img_to_array(img)

x = x.reshape(1, 150, 150, 3)

i = 0

for batch in datagen.flow(x, batch_size=1):
	plt.figure(i)	#第i个图像
	i += 1
	imgplot = plt.imshow(image.array_to_img(batch[0]))
	if i % 4 ==0:
		break

plt.show()

	
