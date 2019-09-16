import cv2
import numpy as np
from keras import models

model = models.load_model("custom_sign_model_v2.h5")

class_names = ["bird", "boar", "dog", "dragon", "hare", "horse", "monkey", "ox", "ram", "rat", "serpant", "tiger", "none"]

cap = cv2.VideoCapture(0)
x, y, w, h = 490, 250, 350, 350 # roi crop

while(True):
    ret, frame = cap.read()
    if ret:
        draw = frame.copy()
        cv2.rectangle(draw, (x, y), (x+w, y+h), (255, 255, 0), 2)
        cv2.imshow("webcam", draw)

        image = frame[y:y+w, x:x+w]

        resized = cv2.resize(image, (256, 256), interpolation = cv2.INTER_AREA)
        rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)/255.
        input = np.expand_dims(rgb_img, axis=0)
        output = model.predict(input)

        print(class_names[np.argmax(output)])
        # cv2.waitKey(250)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
