import cv2
import time
import numpy as np
from keras import models


def draw_label(frame, text, pos, font_scale, text_color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
    # set the text start position
    text_offset_x, text_offset_y = pos
    # make the coords of the box with a small padding of two pixels
    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width - 2, text_offset_y - text_height - 2))
    cv2.rectangle(frame, box_coords[0], box_coords[1], (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=text_color, thickness=2)
    return frame


class NarutoGame(object):
    def __init__(self):
        # Using OpenCV to capture from device 0 and read the first frame
        self.video = cv2.VideoCapture(0)

        # Load Keras model for hand sign classification
        self.model = models.load_model("../data/custom_sign_model_v3.h5")
        self.class_names = ["bird", "boar", "dog", "dragon", "hare", "horse", "monkey", "ox", "ram", "rat", "serpant", "tiger", "none"]
        self.jutsus = {"11,0": "shuriken jutsu",
                       "5,10": "fire ball jutsu"}

        self.prev_sign = 12
        self.sign_count = 0
        self.curr_combo = []
        self.clear_count = 0
        self.time_start = time.time()
        self.player_switch = False

        self.game_round = 1
        self.curr_player = 1
        self.player_stats = [{"hp": 100, "chakra": 100, "specials": False, "attack": -1},
                             {"hp": 100, "chakra": 100, "specials": False, "attack": -1}]

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()
        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX

        if ret:
            # Show player roi region
            x, y, w, h = (width//2-width//8, height//3, width//4, width//4)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

            # draw player stats
            frame = draw_label(frame, f"Round {self.game_round}", (30, 30), 1, (255,255,0))
            frame = draw_label(frame, f"Player {self.curr_player}", (int(width*0.7), int(height*0.8)), 1.3, (255,255,0))
            frame = draw_label(frame, f"HP: {self.player_stats[self.curr_player-1]['hp']}",(int(width*0.7), int(height*0.85)), 1.1, (255,255,0))
            frame = draw_label(frame, f"Chakra: {self.player_stats[self.curr_player-1]['chakra']}",(int(width*0.7), int(height*0.9)), 1.1, (255,255,0))
            frame = draw_label(frame, f"Specials: {self.player_stats[self.curr_player-1]['specials']}",(int(width*0.7), int(height*0.95)), 1.1, (255,255,0))

            if self.player_switch:
                if self.game_round >=3:
                    self.player_stats[0]['specials'] = True
                    self.player_stats[1]['specials'] = True

                cv2.putText(frame, "Player switch!!",(x, int(height*0.9)), font, 1.5,(255,255,0), 2)
                # 10s count down
                time_elapse = int(time.time() - self.time_start)
                cv2.putText(frame, f"{10 - time_elapse}",(width//2, 60), font, 2,(0,0,255), 2)

                if time_elapse >= 10:
                    self.game_round += 1
                    self.player_switch = False
                    self.time_start = time.time()

            else:
                roi = frame[y:y+w, x:x+w]
                resized_input = cv2.resize(roi, (256, 256), interpolation = cv2.INTER_AREA)
                rgb_img = cv2.cvtColor(resized_input, cv2.COLOR_BGR2RGB)/255.
                input = np.expand_dims(rgb_img, axis=0)
                output = self.model.predict(input)

                # Show recongized hand sign
                curr_sign = np.argmax(output)
                cv2.putText(frame, f"Detected sign: {self.class_names[curr_sign]}",(x, y), font, 1,(0,255,0), 2)

                # 20s count down
                time_elapse = int(time.time() - self.time_start)
                cv2.putText(frame, f"{20 - time_elapse}",(width//2, 60), font, 2,(0,0,255), 2)

                if time_elapse >= 20:
                    if self.curr_player == 1:
                        self.curr_player = 2
                    else:
                        self.curr_player = 1
                    self.player_switch = True
                    self.time_start = time.time()

                # Combo tracking
                if curr_sign == self.prev_sign:
                    self.sign_count += 1

                if self.sign_count > 8:
                    if curr_sign == 12 and self.curr_combo != []:
                        self.clear_count += 1
                    elif str(curr_sign) not in self.curr_combo and curr_sign != 12:
                        self.curr_combo.append(str(curr_sign))
                    self.sign_count = 0

                combo_text = [self.class_names[int(num)] for num in self.curr_combo]
                frame = draw_label(frame, "Combo: "+"->".join(combo_text),(0, int(height*0.9)), 1.2,(255,255,0))

                # Clear combo or release attack to end turn early
                if self.clear_count > 1 and self.curr_combo != []:
                    jutsu = ",".join(self.curr_combo)
                    if jutsu in self.jutsus.keys():
                        print(f"{self.jutsus[jutsu]}!!")
                        cv2.putText(frame, f"{self.jutsus[jutsu]}!!",(frame.shape[1]//3, int(frame.shape[0]*0.9)), font, 1.5,(255,255,0), 2)
                        # Update stats
                        self.player_stats[self.curr_player-1]["chakra"] -= len(self.curr_combo)*10
                        if self.curr_player == 1:
                            self.curr_player = 2
                        else:
                            self.curr_player = 1
                        self.player_stats[self.curr_player-1]["hp"] -= len(self.curr_combo)*10
                        self.player_switch = True
                        self.time_start = time.time()

                    self.curr_combo = []
                    self.clear_count = 0

                self.prev_sign = curr_sign

        return frame

if __name__ == '__main__':
    new_game = NarutoGame()
    while True:
        frame = new_game.get_frame()
        cv2.imshow("Webcam frame", frame)

        if cv2.waitKey(1) == 27:
            break  # esc to quit

    cv2.destroyAllWindows()
