import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# =========================
# DATASET PATH
# =========================

dataset_path = "dataset"

# =========================
# IMAGE SIZE
# =========================

img_height = 28
img_width = 28
batch_size = 32

# =========================
# DATA AUGMENTATION
# =========================

datagen = ImageDataGenerator(

    rescale=1./255,

    validation_split=0.2,

    rotation_range=15,

    zoom_range=0.15,

    width_shift_range=0.15,

    height_shift_range=0.15

)

# =========================
# TRAIN DATASET
# =========================

train_dataset = datagen.flow_from_directory(

    dataset_path,

    target_size=(img_height, img_width),

    batch_size=batch_size,

    color_mode='grayscale',

    class_mode='sparse',

    subset='training'

)

# =========================
# VALIDATION DATASET
# =========================

validation_dataset = datagen.flow_from_directory(

    dataset_path,

    target_size=(img_height, img_width),

    batch_size=batch_size,

    color_mode='grayscale',

    class_mode='sparse',

    subset='validation'

)

# =========================
# CNN MODEL
# =========================

model = models.Sequential()

model.add(layers.Input(shape=(28,28,1)))

model.add(layers.Conv2D(
    32,
    (3,3),
    activation='relu'
))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(
    64,
    (3,3),
    activation='relu'
))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Dropout(0.25))

model.add(layers.Flatten())

model.add(layers.Dense(
    128,
    activation='relu'
))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(
    10,
    activation='softmax'
))

# =========================
# COMPILE
# =========================

model.compile(

    optimizer='adam',

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy']

)

# =========================
# TRAIN
# =========================

history = model.fit(

    train_dataset,

    validation_data=validation_dataset,

    epochs=20

)

# =========================
# SAVE MODEL
# =========================

model.save("digit_model.keras")

# =========================
# CREATE PLOTS FOLDER
# =========================

os.makedirs("plots", exist_ok=True)

# =========================
# ACCURACY GRAPH
# =========================

plt.figure(figsize=(8,5))

plt.plot(
    history.history['accuracy'],
    label='Train Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.title('Accuracy Graph')

plt.legend()

plt.savefig("plots/accuracy.png")

# =========================
# LOSS GRAPH
# =========================

plt.figure(figsize=(8,5))

plt.plot(
    history.history['loss'],
    label='Train Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.title('Loss Graph')

plt.legend()

plt.savefig("plots/loss.png")

plt.show()

# =========================
# PRINT CLASS INDEX
# =========================

print("\nClass Index:")
print(train_dataset.class_indices)