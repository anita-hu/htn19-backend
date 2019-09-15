import cv2
import time
import numpy as np
from keras.models import load_model


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
    def __init__(self, voice=True):
        self.video = cv2.VideoCapture(0)

        self.model = load_model("data/custom_sign_model_v3.h5")
        self.class_names = ["bird", "boar", "dog", "dragon", "hare", "horse", "monkey", "ox", "ram", "rat", "serpant", "tiger", "none"]
        self.jutsu_names = ["Start Game", "Lightning Dragon Jutsu", "Fire Ball Jutsu", "Rasengan", "Rasenshuriken", "Water Shark"]
        self.jutsus = {"2,9": 3, "8,2": 5, "5,10": 2, "2,10,3": 1, "5,6,11": 4}

        self.use_voice = voice
        self.prev_sign = 12
        self.sign_count = 0
        self.curr_combo = []
        self.clear_count = 0
        self.time_start = time.time()
        self.player_switch = False
        self.player_special_used = [False, False]

        self.game_round = 1
        self.curr_player = 1
        self.player_stats = [{"hp": 100, "chakra": 100, "specials": 0, "attack": -1},
                             {"hp": 100, "chakra": 100, "specials": 0, "attack": -1}]

    def __del__(self):
        self.video.release()

    @staticmethod
    def detect_cards(frame, curr_player):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if curr_player == 1:
            # Orange card
            lower_hsv, higher_hsv = np.array([5, 141, 120]), np.array([ 16, 253, 255])
            org_mask = cv2.inRange(hsv, lower_hsv, higher_hsv)

            # Yellow card
            lower_hsv, higher_hsv = np.array([17, 147, 123]), np.array([ 70, 253, 255])
            yel_mask = cv2.inRange(hsv, lower_hsv, higher_hsv)

            mask = cv2.bitwise_or(org_mask, yel_mask) # naruto
        else:
            # Blue card
            lower_hsv, higher_hsv = np.array([ 84,  58, 127]), np.array([129, 253, 255])
            blue_mask = cv2.inRange(hsv, lower_hsv, higher_hsv)

            # Red card
            lower_hsv, higher_hsv = np.array([132, 72, 150]), np.array([180, 253, 255])
            red_mask1 = cv2.inRange(hsv, lower_hsv, higher_hsv)

            # Red card
            lower_hsv, higher_hsv = np.array([0, 124, 161]), np.array([5, 253, 255])
            red_mask2 = cv2.inRange(hsv, lower_hsv, higher_hsv)

            mask = cv2.bitwise_or(blue_mask, red_mask1) # sasuke
            mask = cv2.bitwise_or(mask, red_mask2)

        overlay = cv2.resize(mask, (frame.shape[1]//3, frame.shape[0]//3))
        x_offset = frame.shape[1]-overlay.shape[1]
        frame[0:overlay.shape[0], x_offset:x_offset+overlay.shape[1], 0] = overlay
        frame[0:overlay.shape[0], x_offset:x_offset+overlay.shape[1], 1] = overlay
        frame[0:overlay.shape[0], x_offset:x_offset+overlay.shape[1], 2] = overlay

        return cv2.countNonZero(mask) > 500

    def get_frame(self, meta):
        ret, frame = self.video.read()
        font = cv2.FONT_HERSHEY_SIMPLEX
        start_game = meta["start"]
        voice_sign = meta["sign"]
        end_game = meta["end"]
        restart = meta["restart"]

        if ret:
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            # Show player roi region
            x, y, w, h = (width//2-width//8, height//3, width//4, width//4)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

            if not start_game:
                frame = draw_label(frame, "Waiting for start command", (30, 30), 1, (255,255,0))
            elif end_game:
                frame = draw_label(frame, "End game", (30, 30), 1, (255,255,0))
            elif restart:
                self.prev_sign = 12
                self.sign_count = 0
                self.curr_combo = []
                self.clear_count = 0
                self.time_start = time.time()
                self.player_switch = False
                self.game_round = 1
                self.curr_player = 1
                self.player_special_used = [False, False]
                self.player_stats = [{"hp": 100, "chakra": 100, "specials": 0, "attack": -1},
                                     {"hp": 100, "chakra": 100, "specials": 0, "attack": -1}]
            else:
                # Draw player stats
                frame = draw_label(frame, f"Round {self.game_round}", (0, 30), 1.3, (255,255,0))
                frame = draw_label(frame, "Naruto", (int(width*0.7), int(height*0.55)), 1.3, (255,255,0))
                frame = draw_label(frame, f"HP: {self.player_stats[0]['hp']}",(int(width*0.7), int(height*0.6)), 1.1, (255,255,0))
                frame = draw_label(frame, f"Chakra: {self.player_stats[0]['chakra']}",(int(width*0.7), int(height*0.65)), 1.1, (255,255,0))
                frame = draw_label(frame, f"Specials: {self.player_stats[0]['specials']}",(int(width*0.7), int(height*0.7)), 1.1, (255,255,0))

                frame = draw_label(frame, "Sasuke", (int(width*0.7), int(height*0.8)), 1.3, (255,255,0))
                frame = draw_label(frame, f"HP: {self.player_stats[1]['hp']}",(int(width*0.7), int(height*0.85)), 1.1, (255,255,0))
                frame = draw_label(frame, f"Chakra: {self.player_stats[1]['chakra']}",(int(width*0.7), int(height*0.9)), 1.1, (255,255,0))
                frame = draw_label(frame, f"Specials: {self.player_stats[1]['specials']}",(int(width*0.7), int(height*0.95)), 1.1, (255,255,0))

                if self.detect_cards(frame, self.curr_player) and self.player_stats[self.curr_player-1]['specials'] == 100:
                    print("Using special")
                    cv2.putText(frame, "Special attack!!", (x, int(height*0.9)), font, 1.5,(255,255,0), 2)
                    self.player_special_used[self.curr_player-1] = True
                    self.player_stats[self.curr_player-1]["specials"] = 0
                    self.player_stats[self.curr_player-1]["attack"] = 8
                    # Update stats
                    if self.curr_player == 1:
                        self.curr_player = 2
                    else:
                        self.curr_player = 1
                    self.player_stats[self.curr_player-1]["hp"] -= 40
                    self.player_switch = True
                    self.time_start = time.time()
                else:
                    if self.player_switch:
                        cv2.putText(frame, "Player switch!!",(x, int(height*0.9)), font, 1.5,(255,255,0), 2)
                        # 10s count down
                        time_elapse = int(time.time() - self.time_start)
                        cv2.putText(frame, f"{10 - time_elapse}",(width//2, 60), font, 2,(0,0,255), 2)

                        if time_elapse >= 10:
                            self.game_round += 1
                            self.player_switch = False
                            self.time_start = time.time()
                            self.player_stats[0]['attack'] = -1
                            self.player_stats[1]['attack'] = -1
                            self.curr_combo = []
                            self.sign_count = 0
                            self.prev_sign = 12
                            self.clear_count = 0

                            if self.game_round < 3:
                                self.player_stats[0]['specials'] += 50
                                self.player_stats[1]['specials'] += 50

                            elif self.game_round >= 3:
                                self.player_stats[0]['specials'] = 0
                                self.player_stats[1]['specials'] = 0
                                if not self.player_special_used[0]:
                                    self.player_stats[0]['specials'] = 100
                                elif not self.player_special_used[1]:
                                    self.player_stats[1]['specials'] = 100

                    else:
                        roi = frame[y:y+w, x:x+w]
                        roi = cv2.flip(roi, 1)
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

                        if self.sign_count > 5:
                            if curr_sign == 12 and self.curr_combo != []:
                                self.clear_count += 1
                            elif str(curr_sign) not in self.curr_combo and curr_sign != 12:
                                self.curr_combo.append(str(curr_sign))
                            self.sign_count = 0

                        combo_text = [self.class_names[int(num)] for num in self.curr_combo]
                        frame = draw_label(frame, "Combo: "+"->".join(combo_text),(0, int(height*0.9)), 1.2,(255,255,0))

                        # Clear combo or release attack to end turn early

                        if self.clear_count > 1:
                            self.curr_combo = []
                            self.clear_count = 0

                        if self.curr_combo != []:
                            jutsu = ",".join(self.curr_combo)
                            passed = False
                            if (self.use_voice and self.jutsus[jutsu] == voice_sign) or not self.use_voice:
                                passed = True

                            if jutsu in self.jutsus.keys() and passed:
                                print(f"{self.jutsus[jutsu]}!!")
                                cv2.putText(frame, f"{self.jutsu_names[self.jutsus[jutsu]]}!!",(frame.shape[1]//3, int(frame.shape[0]*0.9)), font, 1.5,(255,255,0), 2)
                                self.player_stats[self.curr_player-1]['attack'] = self.jutsus[jutsu]
                                # Update stats
                                self.player_stats[self.curr_player-1]["chakra"] -= len(self.curr_combo)*10
                                if self.curr_player == 1:
                                    self.curr_player = 2
                                else:
                                    self.curr_player = 1
                                self.player_stats[self.curr_player-1]["hp"] -= len(self.curr_combo)*10
                                self.player_switch = True
                                self.time_start = time.time()

                        self.prev_sign = curr_sign

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
