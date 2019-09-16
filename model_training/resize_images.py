import cv2
import os

images = os.listdir("train")
input_size = 256

for image_file in images:
    if "jpg" in image_file:
        image = cv2.imread(os.path.join("train", image_file), 1)
        resized = cv2.resize(image, (input_size, input_size), interpolation = cv2.INTER_AREA)
        cv2.imwrite(os.path.join("train", image_file), resized)

        print(os.path.join("train", image_file))
        print(resized.shape)
