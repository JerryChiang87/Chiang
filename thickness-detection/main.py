import keras
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


train_data_dir="data_3categories/train"

datagen = ImageDataGenerator(rescale=1. / 255)
generator = datagen.flow_from_directory(
       train_data_dir,
       target_size=(224, 224),
       batch_size=22,
       class_mode="categorical",
       shuffle=True)
b = np.zeros((generator.classes.size, generator.classes.max()+1))
b[np.arange(generator.classes.size),generator.classes] = 1
b = b.astype("int64")
print(generator.classes)
print(np.argmax(generator, axis=1))
#y = generator.next()
#print(y.type)