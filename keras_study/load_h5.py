from keras.models import load_model

model_dir = '/home/wildchap/python/keras_study/cats_and_dogs_small_1.h5'
model = load_model(model_dir)
model.summary()
