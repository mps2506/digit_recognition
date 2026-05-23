import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load model
model = tf.keras.models.load_model("digit_model.h5")

# Đọc ảnh
img = cv2.imread("test_images/5.png", cv2.IMREAD_GRAYSCALE)

# Resize về 28x28
img = cv2.resize(img, (28, 28))

# Đảo màu nếu cần
img = 255 - img

# Chuẩn hóa
img = img / 255.0

# Reshape
img = img.reshape(1, 28, 28, 1)

# Predict
prediction = model.predict(img)

digit = np.argmax(prediction)

print("Số dự đoán:", digit)

# Hiển thị ảnh
plt.imshow(img.reshape(28,28), cmap='gray')
plt.title(f"Predict: {digit}")
plt.show()