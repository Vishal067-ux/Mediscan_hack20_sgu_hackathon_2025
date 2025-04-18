from google.colab import drive
drive.mount('/content/drive')

import os

import os
import pandas as pd

# Correct base paths
base_path = "/content/drive/MyDrive/datast"
metadata_path = os.path.join(base_path, "HAM10000_metadata.csv")

# Image folders
image_folders = [
    os.path.join(base_path, "HAM10000_images_part_1"),
    os.path.join(base_path, "HAM10000_images_part_2")
]

import cv2
import numpy as np

def load_images_from_drive(image_paths, img_size=(64, 64)):
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, img_size)
            images.append(img)
    return np.array(images)

X = load_images_from_drive(df['image_path'])

import cv2
import numpy as np

def load_images_from_drive(image_paths, img_size=(64, 64)):
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            img = cv2.resize(img, img_size)
            images.append(img)
    return np.array(images)

X = load_images_from_drive(df['image_path'])
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf

df['label'] = LabelEncoder().fit_transform(df['dx'])
y = tf.keras.utils.to_categorical(df['label'])

X = X / 255.0

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)




from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
   
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(y.shape[1], activation='softmax')  # Output layer
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, y_val)
)

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.show()

!MyDrive/datast/second_skin_disease_model.h5

try:
    model.save("/content/drive/MyDrive/datast/skin_disease_model.h5")
    print("✅ Model saved successfully.")
except Exception as e:
    print("❌ Error saving model:", e)
