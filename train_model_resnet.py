import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# =========================
# LOAD DATA
# =========================

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize
train_images = train_images / 255.0
test_images = test_images / 255.0

# Resize 32x32
train_images = tf.image.resize(
    train_images[..., tf.newaxis],
    (32, 32)
)

test_images = tf.image.resize(
    test_images[..., tf.newaxis],
    (32, 32)
)

# Convert grayscale -> RGB
train_images = tf.image.grayscale_to_rgb(train_images)
test_images = tf.image.grayscale_to_rgb(test_images)

# =========================
# DATA AUGMENTATION
# =========================

datagen = ImageDataGenerator(

    rotation_range=25,

    zoom_range=0.25,

    width_shift_range=0.25,

    height_shift_range=0.25

)

# =========================
# LOAD RESNET
# =========================

base_model = tf.keras.applications.ResNet50(

    include_top=False,

    weights=None,

    input_shape=(32,32,3),

    pooling='avg'

)

# =========================
# BUILD MODEL
# =========================

model = models.Sequential([

    base_model,

    layers.Dense(128, activation='relu'),

    layers.Dropout(0.5),

    layers.Dense(10, activation='softmax')

])

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

    datagen.flow(
        train_images,
        train_labels,
        batch_size=64
    ),

    epochs=10,

    validation_data=(
        test_images,
        test_labels
    )

)

# =========================
# SAVE
# =========================

model.save("resnet_digit.keras")

# =========================
# EVALUATE
# =========================

test_loss, test_acc = model.evaluate(
    test_images,
    test_labels
)

print(f"\nAccuracy: {test_acc*100:.2f}%")

# =========================
# PLOT
# =========================

plt.plot(
    history.history['accuracy'],
    label='Train Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.legend()

plt.show()