from flask import Flask, render_template, Response, jsonify
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
    # setup hand result
    right_hand_result = "None"
    left_hand_result = "None"

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
                        #print('Right')
                        right_hand_result = resutl_name
                        #print(right_hand_result)
                    else:
                        #print('Left')
                        left_hand_result = resutl_name
                        #print(left_hand_result)
                    # draw handlandmarks on image:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())

                image = cv2.flip(image, 1)
                h, w = image.shape[0], image.shape[1]
                image = cv2.putText(image, right_hand_result, (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(40,44,52), )
                image = cv2.putText(image, left_hand_result, (w-30, h-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(40,44,52))

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


@app.route('/results', methods=['GET'])
def results():
    text_right = right_hand_result
    text_left = left_hand_result
    return jsonify(result=[text_left, text_right])


# @app.route('/results_left')
# def results_left():
#     return left_hand_result
#
#
# @app.route('/results_right')
# def results_right():
#     return right_hand_result

# @app.route('/results', methods=['POST'])
# def results():
#     return render_template("index.html", left_result=left_hand_result, right_result=right_hand_result)


if __name__ == "__main__":
    app.run()
