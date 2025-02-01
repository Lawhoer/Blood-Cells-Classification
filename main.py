# Import libraries
import tensorflow as tf
from keras._tf_keras import keras
from keras._tf_keras.keras import utils
from keras._tf_keras.keras import layers
import os
import pandas as pd
import numpy as np
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

sns.set_theme(style="ticks")

path = os.path.join("data/bloodcells_dataset")

BATCH_SIZE = 16
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Load the dataset
dataset, test_dataset = utils.image_dataset_from_directory(directory=path,
                                                                 image_size=(IMG_WIDTH, IMG_HEIGHT),
                                                                 batch_size=BATCH_SIZE,
                                                                 label_mode="int",
                                                                 validation_split=0.2,
                                                                 subset="both",
                                                                 shuffle=True,
                                                                 seed=5)


"""
# Class names
class_names = dataset.class_names
print(class_names)

# Visualize the dataset
fig, ax = plt.subplots(4, 4, figsize=(12,12))
ax = ax.flat 
for images, labels in dataset.take(1):
  for i in range(16):
    ax[i].set_title(class_names[labels[i].numpy()])
    ax[i].set_xticks([]) 
    ax[i].set_yticks([])
    ax[i].imshow(images[i].numpy().astype("uint8"))

plt.tight_layout()
plt.show()


# Class distribution
labels = np.concatenate([label for image, label in dataset], axis=0)
unique, counts = np.unique(labels, return_counts=True)
plt.pie(x=counts, labels=class_names, autopct='%.1f%%', textprops={'size': 'smaller'},
        colors=sns.color_palette('pastel')[0:8])

plt.title("Class distribution")
plt.show() 
"""

# Split the dataset into training and validation
num_elements = len(dataset)
train_size = int(0.8 * num_elements)
val_dataset = dataset.skip(train_size).prefetch(tf.data.AUTOTUNE)
train_dataset = dataset.take(train_size).prefetch(tf.data.AUTOTUNE)

""" 
# Test dataset size
len(train_dataset), len(val_dataset), len(test_dataset)
print(f"Training dataset size: {len(train_dataset)} batches")
print(f"Validation dataset size: {len(val_dataset)} batches")
print(f"Test dataset size: {len(test_dataset)} batches")

for image_batch, labels_batch in train_dataset.take(1):
  print(f"Train data: {image_batch.shape}")
  print(f"Train labels: {labels_batch.shape}")

for image_batch, labels_batch in val_dataset.take(1):
  print(f"Validation data: {image_batch.shape}")
  print(f"Validation labels: {labels_batch.shape}")

for image_batch, labels_batch in test_dataset.take(1):
  print(f"Test data: {image_batch.shape}")
  print(f"Test labels: {labels_batch.shape}")


# Check the range of pixel values
for image, label in train_dataset.take(1):
    print(tf.reduce_max(image))
    print(tf.reduce_min(image))
"""

lab_book = {}

early_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True,
                                         monitor="val_accuracy", min_delta=0.005)

data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal_and_vertical"),
    keras.layers.RandomRotation(0.4)
    ])


name = "Model_4"

# Build model
tf.random.set_seed(5)
model_2 = keras.Sequential(
    [layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
     layers.Rescaling(1./255),
     data_augmentation, # 2
     layers.Conv2D(filters=8, kernel_size=(3,3),
                   padding="same", activation="relu",
                   kernel_initializer="he_normal"),
     layers.BatchNormalization(), #3
     layers.MaxPool2D(),

     layers.Conv2D(filters=8, kernel_size=(3,3), #3
                   padding="same", activation="relu",
                   kernel_initializer="he_normal"),
     layers.BatchNormalization(), #3
     layers.MaxPool2D(), #3

     layers.Conv2D(filters=8, kernel_size=(3,3),  #3
                   padding="same", activation="relu",
                   kernel_initializer="he_normal"),
     layers.BatchNormalization(), #3
     layers.MaxPool2D(), #3



     layers.Flatten(),
     layers.Dense(112,activation='relu'), #3
     layers.Dropout(0.5), #3
     layers.Dense(112,activation='relu'), #3
     layers.Dropout(0.5), #3
     layers.Dense(8, activation="softmax")
     ], name=name)

model_2.summary()

# Compile model
model_2.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
                optimizer=keras.optimizers.Adam(),
                metrics=["accuracy"])

# Fit model
history_2 = model_2.fit(train_dataset,
                        validation_data=val_dataset,
                        epochs=50,
                        callbacks=[early_cb])


# Write lab-book
train_accuracy = model_2.evaluate(train_dataset)[1]
val_accuracy = model_2.evaluate(val_dataset)[1]
lab_book[name] = {"train_accuracy": train_accuracy, "val_accuracy": val_accuracy}

model_2.save("More_Complex.h5")