import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

# Tạo thư mục dataset
for i in range(10):
    os.makedirs(f"dataset/{i}", exist_ok=True)

# Font
font = ImageFont.truetype("arial.ttf", 80)

# Sinh dữ liệu
samples_per_class = 1000

for digit in range(10):

    for index in range(samples_per_class):

        # Canvas đen
        img = Image.new("L", (100,100), color=0)

        draw = ImageDraw.Draw(img)

        # Random vị trí
        x = random.randint(20,40)
        y = random.randint(5,20)

        # Vẽ số
        draw.text(
            (x,y),
            str(digit),
            fill=255,
            font=font
        )

        # Convert numpy
        img_np = np.array(img)

        # Random rotation
        angle = random.randint(-20,20)

        matrix = cv2.getRotationMatrix2D(
            (50,50),
            angle,
            1
        )

        img_np = cv2.warpAffine(
            img_np,
            matrix,
            (100,100)
        )

        # Resize 28x28
        img_np = cv2.resize(img_np, (28,28))

        # Save
        cv2.imwrite(
            f"dataset/{digit}/{index}.png",
            img_np
        )

print("Dataset generated!")