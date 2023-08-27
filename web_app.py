from flask import Flask, render_template, Response
import cv2
from logic import HandClassifierHandler

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


# setup Flask app
app = Flask(__name__)
camera = cv2.VideoCapture(0)

# Read model
hch = HandClassifierHandler.HandClassifierHandler()
model = hch.load_model()

def get_frames():
    with mp_hands.Hands(
            model_complexity=0,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7) as hands:

        while True:
            success, image = camera.read()
            if not success:
                break
            else:
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)
                # Draw the hand annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    message = results.multi_handedness
                    # obtain result from classificator:
                    result = hch.get_result(model=model, handlandmarks=results.multi_hand_landmarks[0],
                                            is_R=hch.is_right(message))
                    resutl_name = hch.result_parser(result=result)
                    # check which hand:
                    if hch.is_right(message):
                        print('Right')
                        #detect_left_label.config(text=resutl_name)
                    else:
                        print('Left')
                        #detect_right_label.config(text=resutl_name)
                    # draw handlandmarks on image:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())

                image = cv2.flip(image, 1)
                ret, buffer = cv2.imencode('.jpg', image)
                image = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(get_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":

    app.run()
