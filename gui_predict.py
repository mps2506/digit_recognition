import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import cv2
import tensorflow as tf

# =========================
# LOAD MODEL
# =========================

model = tf.keras.models.load_model("digit_model.keras")

# =========================
# TẠO CỬA SỔ
# =========================

window = tk.Tk()
window.title("Digit Recognition")

canvas_width = 280
canvas_height = 280

# =========================
# CANVAS
# =========================

canvas = tk.Canvas(
    window,
    width=canvas_width,
    height=canvas_height,
    bg='black'
)

canvas.grid(row=0, column=0, columnspan=2)

# =========================
# PIL IMAGE
# =========================

image = Image.new("L", (canvas_width, canvas_height), color=0)
draw = ImageDraw.Draw(image)

# =========================
# LABEL KẾT QUẢ
# =========================

label_result = tk.Label(
    window,
    text="Dự đoán: ",
    font=("Arial", 20)
)

label_result.grid(row=1, column=0, columnspan=2)

# =========================
# BIẾN VẼ
# =========================

last_x = None
last_y = None

# =========================
# HÀM VẼ
# =========================

def draw_lines(event):

    global last_x, last_y

    x, y = event.x, event.y

    if last_x is not None and last_y is not None:

        # Vẽ lên canvas GUI
        canvas.create_line(
            last_x,
            last_y,
            x,
            y,
            fill='white',
            width=12,
            capstyle=tk.ROUND,
            smooth=True
        )

        # Vẽ lên ảnh PIL
        draw.line(
            [last_x, last_y, x, y],
            fill=255,
            width=12
        )

    last_x = x
    last_y = y

# =========================
# RESET CHUỘT
# =========================

def reset(event):

    global last_x, last_y

    last_x = None
    last_y = None

# =========================
# GẮN EVENT
# =========================

canvas.bind("<B1-Motion>", draw_lines)
canvas.bind("<ButtonRelease-1>", reset)

# =========================
# PREDICT
# =========================

def predict_digit():

    # PIL -> numpy
    img = np.array(image)

    # Threshold
    _, thresh = cv2.threshold(
        img,
        50,
        255,
        cv2.THRESH_BINARY
    )

    # Find contours
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:

        label_result.config(
            text="Không có số"
        )

        return

    # Contour lớn nhất
    contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(contour)

    # Crop digit
    digit = thresh[y:y+h, x:x+w]

    # Resize giữ tỉ lệ
    h_digit, w_digit = digit.shape

    if h_digit > w_digit:

        new_h = 20
        new_w = int(w_digit * (20 / h_digit))

    else:

        new_w = 20
        new_h = int(h_digit * (20 / w_digit))

    digit = cv2.resize(
        digit,
        (new_w, new_h)
    )

    # Tạo canvas 28x28
    final_img = np.zeros(
        (28, 28),
        dtype=np.uint8
    )

    # Center digit
    # Tính center of mass
    coords = np.column_stack(np.where(digit > 0))

    center_y, center_x = coords.mean(axis=0)

    # Tạo ảnh 28x28
    final_img = np.zeros((28, 28), dtype=np.uint8)

    # Offset căn giữa
    x_offset = int(14 - center_x)
    y_offset = int(14 - center_y)

    # Resize digit vào giữa
    for y in range(digit.shape[0]):
        for x in range(digit.shape[1]):

            new_x = x + x_offset
            new_y = y + y_offset

            if 0 <= new_x < 28 and 0 <= new_y < 28:
                final_img[new_y, new_x] = digit[y, x]

    final_img[
        y_offset:y_offset+new_h,
        x_offset:x_offset+new_w
    ] = digit

    # Normalize
    final_img = final_img / 255.0

    # Reshape
    final_img = final_img.reshape(
        1,
        28,
        28,
        1
    )

    # Predict
    prediction = model.predict(final_img)

    predicted_digit = np.argmax(prediction)

    confidence = np.max(prediction) * 100

    # Hiển thị kết quả
    label_result.config(
        text=f"Dự đoán: {predicted_digit} ({confidence:.2f}%)"
    )

# =========================
# CLEAR
# =========================

def clear_canvas():

    global image, draw

    canvas.delete("all")

    image = Image.new(
        "L",
        (canvas_width, canvas_height),
        color=0
    )

    draw = ImageDraw.Draw(image)

    label_result.config(
        text="Dự đoán: "
    )

# =========================
# BUTTON PREDICT
# =========================

btn_predict = tk.Button(
    window,
    text="Predict",
    command=predict_digit,
    width=15,
    height=2
)

btn_predict.grid(row=2, column=0)

# =========================
# BUTTON CLEAR
# =========================

btn_clear = tk.Button(
    window,
    text="Clear",
    command=clear_canvas,
    width=15,
    height=2
)

btn_clear.grid(row=2, column=1)

# =========================
# MAIN LOOP
# =========================

window.mainloop()