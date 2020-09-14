import numpy as np
import matplotlib.pyplot as plt
from keras import models
from keras.preprocessing import image

#单张图片路径
img_path = '/home/wildchap/cats_and_dogs_small/test/cats/cat.1978.jpg'

#将图像预处理为一个4D张量
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
print(img_tensor.shape)	#(150, 150, 3)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.0
print(img_tensor.shape)	#(1, 150, 150, 3)

#显示原图像
plt.imshow(img_tensor[0])	#由于只有一张图
plt.show()

#提取之前训练的模型
model = models.load_model('cats_and_dogs_small_1.h5')

#输入一张图像，模型返回原始模型前8层的激活值
#提取前8层的输出
layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

#predict返回8个Numpy数组，来反映每一层的处理情况
activations = activation_model.predict(img_tensor)

#第一层卷积激活后的数据
first_layer_activation = activations[0]
print(first_layer_activation.shape)	#(1, 148, 148, 32)	第一张图，148*148的特征图，有32个通道

#将第6个通道可视化
plt.matshow(first_layer_activation[0, :, :, 6], cmap='viridis')
plt.show()

#将前8层的名称存储到数组中
layer_names = []
for layer in model.layers[:8]:
	layer_names.append(layer.name)
	
#显示图像每一行多少个通道
img_per_row = 16

#将 (层名, 经过该层的特征图矩阵) 打包遍历
for layer_name, layer_activation in zip(layer_names, activations):
	#获取该层通道数(特征数)
	layer_way = layer_activation.shape[-1]
	#特征图的尺寸
	layer_size = layer_activation.shape[1]
	#将通道平铺到一个巨大的图像中
	cols = layer_way // img_per_row		#多少行
	final_pic = np.zeros((cols*layer_size, img_per_row*layer_size))
	
	#写入数值到final_pic
	for col in range(cols):
		for row in range(img_per_row):
			final_pic[col*layer_size: (col+1)*layer_size, row*layer_size: (row+1)*layer_size] \
				= layer_activation[0, :, :, col*img_per_row+row]
	
	scale = 1. / layer_size
	plt.figure(figsize=(scale*final_pic.shape[1], scale*final_pic.shape[0]))
	plt.title(layer_name)
	plt.grid(False)		#不画网格线
	plt.imshow(final_pic, aspect='auto', cmap='viridis')
	plt.show()
