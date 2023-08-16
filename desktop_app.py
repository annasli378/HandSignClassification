import tkinter
import tkinter.font as tkFont

import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
from logic import HandClassifierHandler
from PyQt5 import QtWidgets
import sys

from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *



from tkinter import *
from PIL import Image, ImageTk
import cv2
from tkinter import filedialog

from logic import MainWindow

#app = MainWindow.MainWidnow()
#app.set_window(model=model, hch = hch)



def desktop_app(model, hch):
    # colors here:
    bg_color1 = "#EDEAD0"
    bg_color2 = "#86BAA1"
    # widnow SETTINGS
    win = Tk()
    win.title("PSL Recognition")
    win.geometry("600x500+200+30")
    win.resizable(False, False)
    win.configure(bg=bg_color1)
    w = 400
    h = 300
    frame_1 = Frame(win, width=600, height=320, bg=bg_color2).place(x=0, y=0)
    v = Label(frame_1, width=w, height=h)
    v.place(x=10, y=10)
    cap = cv2.VideoCapture(0)
    # display text:
    about = " Polish Sign Language Recognition Tool  - " \
            "show signs to your device's camera"
    about_box = tkinter.Message(win, anchor="e", bg=bg_color2, justify="center", text=about, font=tkFont.Font(size=12))
    about_box.place(x=420, y=30, width=160, height=160)

    left_label = tkinter.Label(win,  bg=bg_color1, fg="#414361", justify="center", text="Left hand - last detected:", font=tkFont.Font(size=10))
    left_label.place(x=50, y=340, width=180, height=30)
    right_label = tkinter.Label(win,  bg=bg_color1, fg="#414361", justify="center", text="Right hand - last detected:",  font=tkFont.Font(size=10))
    right_label.place(x=400, y=340, width=180, height=30)

    detect_left_label = tkinter.Label(win,  bg=bg_color1, fg="#414361", justify="center", text="None", font=tkFont.Font(size=14))
    detect_left_label.place(x=80, y=400, width=120, height=30)
    detect_right_label = tkinter.Label(win,  bg=bg_color1, fg="#414361", justify="center", text="None",  font=tkFont.Font(size=14))
    detect_right_label.place(x=430, y=400, width=120, height=30)

    def select_img():
        with mp_hands.Hands(
                model_complexity=0,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as hands:
            _, img = cap.read()
            img = cv2.resize(img, (w, h))
            img.flags.writeable = False
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img)
            # Draw the hand annotations on the image:
            img.flags.writeable = True
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                message = results.multi_handedness
                # obtain result from classificator:
                result = hch.get_result(model=model, handlandmarks=results.multi_hand_landmarks[0],
                                        is_R=hch.is_right(message))
                resutl_name = hch.result_parser(result=result)
                # check which hand:
                if hch.is_right(message):
                    print('Right')
                    detect_left_label.config(text=resutl_name)
                else:
                    print('Left')
                    detect_right_label.config(text=resutl_name)
                # draw handlandmarks on image:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        img,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

            # Flip the image horizontally for a selfie-view display: cv2.flip(image, 1)
            # or leave for more video like effect
            imgPIL = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(imgPIL)
            v.configure(image=imgtk)
            v.image = imgtk
            v.after(10, select_img)

    select_img()
    win.mainloop()


def webcam_input(model, hcb):
    # https://github.com/google/mediapipe/blob/master/docs/solutions/hands.md
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
            model_complexity=0,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                message = results.multi_handedness
                print(len(results.multi_hand_landmarks))
                if hch.is_right(message):
                    print('Right')
                    #cv2.putText(image, "Right", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 50, 50), 4)
                else:
                    print('Left')

                result = hch.get_result(model=model, handlandmarks=results.multi_hand_landmarks[0],
                                        is_R=hch.is_right(message))
                # print(results.multi_hand_landmarks)
                # print(((results.multi_hand_landmarks[0]).ListFields()[0][1][0]).ListFields()[2][1])
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())

            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()


########################################################################################################################
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # setup
    hch = HandClassifierHandler.HandClassifierHandler()
    model = hch.load_model()

    desktop_app(model, hch)
