import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob

from sklearn.model_selection import train_test_split
from sklearn import metrics

import cv2
import gc
import os

import tensorflow as tf
import keras
from keras import layers

import warnings
warnings.filterwarnings('ignore')

# Importing Dataset
from zipfile import ZipFile

data_path = 'lung-and-colon-cancer-histopathological-images.zip'

with ZipFile(data_path) as zip:
    zip.extractall()

print('Importing Dataset is completed')


# Data Vizualization
path = 'lung_colon_image_set/lung_image_sets'
classes = os.listdir(path)

for cat in classes:
    image_dir = f'{path}/{cat}'
    images = os.listdir(image_dir)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Images for {cat} category . . . .', fontsize= 20)

    for i in range(3):
        k = np.random.randint(0, len(images))
        img= np.array(Image.open(f'{path}/{cat}/{images[k]}'))
        ax[i].imshow(img)
        ax[i].axis('off')
    plt.show()

print('Data Vizualization is completed')

# Data Preparation
IMG_SIZE = 256
SPLIT = 0.2
EPOCHS = 10
BATCH_SIZE = 64

X= []
Y= []

for i, cat in enumerate(classes):
    images = glob(f'{path}/{cat}/*.jpeg')

    for image in images:
        img = cv2.imread(image)

        X.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
        Y.append(i)

X = np.asarray(X)
one_hot_encoded_Y = pd.get_dummies(Y).values

X_train, X_test, Y_train, Y_test = train_test_split(X, one_hot_encoded_Y, test_size= SPLIT, random_state= 48)

print('Data Preparation is completed')


# Creating Model
model = keras.models.Sequential([
    layers.Conv2D(filters= 32, kernel_size= (5, 5), activation= 'relu', input_shape= (IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(filters= 64, kernel_size= (3, 3), activation= 'relu', padding= 'same'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(filters= 128, kernel_size= (3, 3), activation= 'relu', padding= 'same'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(3, activation= 'softmax')
])

model.compile(optimizer= 'adam', loss='categorical_crossentropy', metrics= ['accuracy'])

print('Creating Model is completed')


# Callbacks
from keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint('output/model_checkpoint.h5',
                             save_best_only= True,
                             verbose=1,
                             save_weights_only= True,
                             monitor= 'val_accuracy')


# Model Training
history = model.fit(X_train, Y_train,
                    validation_data= (X_test, Y_test),
                    batch_size= BATCH_SIZE,
                    epochs= EPOCHS,
                    verbose= 1,
                    callbacks= checkpoint)


# Training Vizualization
history_df = pd.DataFrame(history.history)
history_df.loc[:,['loss','val_loss']].plot()
history_df.loc[:,['accuracy','val_accuracy']].plot()
plt.show()
plt.savefig('output/Training_Results')
