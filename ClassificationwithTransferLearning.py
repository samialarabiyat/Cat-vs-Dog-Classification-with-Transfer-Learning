import os
import zipfile
import random
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
import matplotlib.pyplot as plt

# Download the dataset
!wget https://storage.googleapis.com/tensorflow-1-public/course2/cats_and_dogs_filtered.zip

import os
import zipfile

# Extract the archive
zip_ref = zipfile.ZipFile("./cats_and_dogs_filtered.zip", 'r')
zip_ref.extractall("tmp/")
zip_ref.close()

# Assign training and validation set directories
base_dir = 'tmp/cats_and_dogs_filtered'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# Directory with training cat pictures
train_cats_dir = os.path.join(train_dir, 'cats')

# Directory with training dog pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')

# Directory with validation cat pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')

# Directory with validation dog pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

#https://www.tensorflow.org/tutorials/images/data_augmentation?hl=ar
## Question 1.1 (6 marks) Data Generator
## Use Imagedatagenerator to prepare your training and validation data
## Apply at least 4 techniques for Data Augmentation
# Create a data generator for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Create a data generator for validation data
validation_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches of 32 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# Flow validation images in batches of 32 using validation_datagen generator
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

#https://pythontutorials.eu/deep-learning/image-classification/
## Question 1.2 (5 marks) Prepare your pretrained model without using top layers
## Choose any pretraoned model except Inception model, you must download the weigths and use it
from tensorflow.keras.applications import MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False
base_model.summary()
base_model.trainable = False
base_model.summary()

#https://www.tensorflow.org/guide/keras/sequential_model?hl=ar
## Question 1.5 (6 marks) Build a "" Sequential Model "" using the pretrained model then add the necessary layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

model = Sequential()
model.add(base_model)

model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/Callback
## Question 2.1 (4 marks) Build a callback, use any callback you want except the training accuracy
# Define a Callback class that stops training once accuracy reaches 97.0%
class myCallback(tf.keras.callbacks.Callback):
  def on_train_end(self, logs=None):
    global training_finished
    training_finished = True
    callbacks=[MyCallback()]



model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#https://keras.io/api/callbacks/
## Question 2.2 (3 marks) Training you model using callback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

callbacks = [
    EarlyStopping(patience=2),
    ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    TensorBoard(log_dir='./logs')
]

model.fit(dataset, epochs=10, callbacks=[MyCallback()])

#https://www.tensorflow.org/tutorials/images/transfer_learning?hl=ar
## Question 2.3 (3 marks) Plot the accuracy of the training and validation
import matplotlib.pyplot as plt
history = model2.fit(x_train, y_train, epochs=20,  callbacks=[MyCallback()])
# Assuming history is the object returned from model.fit()
metrics = history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()