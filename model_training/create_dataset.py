import cv2

cap = cv2.VideoCapture(0)
x, y, w, h = 490, 250, 350, 350 # roi crop
count = 449
init = count

while(True):
    ret, frame = cap.read()
    if ret:
        draw = frame.copy()
        cv2.rectangle(draw, (x, y), (x+w, y+h), (255, 255, 0), 2)
        cv2.imshow("webcam", draw)

        image = frame[y:y+w, x:x+w]

        resized = cv2.resize(image, (256, 256), interpolation = cv2.INTER_AREA)
        if count > init + 30: # 30 frames before it starts saving
        #     # change first letter
            cv2.imwrite(f"htn_train/M{count-30}.jpg", resized)

        count += 1
        print(count)
        cv2.waitKey(100)
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= init+150+30: # 150 is the number of images you want to save
        break

cap.release()
cv2.destroyAllWindows()
