import tkinter
import tkinter.font as tkFont
from tkinter import *
from PIL import Image, ImageTk
import cv2
from tkinter import filedialog

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

from logic import HandClassifierHandler




class MainWidnow():
    def __int__(self):
        print("App")

    def set_window(self, model, hch):
        self.model = model
        self.hch = hch
        self.bg_color1 = "#FFA9E7"
        self.bg_color2 = "#FF84E8"
        self.win = Tk()
        self.win.geometry("600x500+200+30")
        self.win.resizable(False, False)
        self.win.configure(bg=self.bg_color1)

        self.w = 400
        self.h = 300

        self.frame = Frame(self.win, width=600, height=320, bg=self.bg_color2).place(x=0, y=0)
        self.v = Label(self.frame, width=400, height=300)
        self.v.place(x=10, y=10)
        self.cap = cv2.VideoCapture(0)


        about = " Polish Sign Language - show sign to camera! "
        self.about_box = tkinter.Message(self.win, anchor="e", bg=self.bg_color1, justify="center", text=about)
        self.about_box.place(x=470, y=30, width=80, height=160)

        self.left_label = tkinter.Label(self.win, bg=self.bg_color2, justify="center", text="Left hand - last detected:",
                                   font=tkFont.Font(size=16))
        self.left_label.place(x=50, y=340, width=136, height=30)
        self.right_label = tkinter.Label(self.win, bg=self.bg_color2, justify="center", text="Right hand - last detected:",
                                    font=tkFont.Font(size=16))
        self.right_label.place(x=400, y=340, width=136, height=30)

        self.detect_left_label = tkinter.Label(self.win, bg=self.bg_color1, justify="center", text="None",
                                          font=tkFont.Font(size=12))
        self.detect_left_label.place(x=50, y=400, width=136, height=30)
        self.detect_right_label = tkinter.Label(self.win, bg=self.bg_color1, justify="center", text="None",
                                           font=tkFont.Font(size=12))
        self.detect_right_label.place(x=400, y=400, width=136, height=30)


        self.select_img(self.cap, self.v, self.model, self.hch)
        self.win.mainloop()

    def select_img(self, cap, v,  model, hch):
        with mp_hands.Hands(
                model_complexity=0,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:
            _, img = cap.read()
            img = cv2.resize(img, (self.w, self.h))
            img.flags.writeable = False
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img)

            # Draw the hand annotations on the image.
            img.flags.writeable = True
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                message = results.multi_handedness
                # check which hand was detected: Right or Left
                if hch.is_right(message):
                    print('Right')
                    # TODO: displaying in text field in app
                else:
                    print('Left')
                # obtain result from classificator
                result = hch.get_result(model=model, handlandmarks=results.multi_hand_landmarks[0],
                                        is_R=hch.is_right(message))
                resutl_name = hch.result_parser(result=result)

                if hch.is_right(message):
                    print('Right')
                    # TODO: displaying in text field in app
                    self.detect_left_label.config(text=resutl_name)


                else:
                    print('Left')
                    self.detect_right_label.config(text=resutl_name)

                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        img,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

            imgPIL = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(imgPIL)
            v.configure(image=imgtk)
            v.image = imgtk
            v.after(10, self.select_img(cap, v,  model, hch))



