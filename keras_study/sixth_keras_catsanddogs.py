import os, shutil	#复制数据集
import numpy as np
import matplotlib.pyplot as plt
from keras import layers, models

#数据集所在路径
datasets_dir = /home/wildchap/cats_and_dogs_small
os.mkdir(datasets_dir)

#区分训练，测试，验证目录
train_dir = os.path.join(datasets_dir, 'train')
test_dir = os.path.join(datasets_dir, 'test')
validation_dir = os.path.join(datasets_dir, 'validation')
os.mkdir(train_dir)
os.mkdir(test_dir)
os.mkdir(validation_dir)

#再分出猫和狗的目录
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(train_cats_dir)
os.mkdir(train_dogs_dir)
os.mkdir(test_cats_dir)
os.mkdir(test_dogs_dir)
os.mkdir(validation_cats_dir)
os.mkdir(validation_dogs_dir)

#猫狗分别取1000个训练数据，500个测试数据和500个验证数据
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
	src
