import pickle
import numpy as np
import mediapipe as mp
mp_hands = mp.solutions.hands

class HandClassifierHandler:
    model = 0
    model_path = ""
    handlandmarks = []
    NUM_CLASSES = 29
    class_nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                   11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                   21, 22, 23, 24, 25, 26, 27, 28]
    class_names = ["0", "1", "2", "3", "4", "5", "A", "B",
                   "C", "D", "E", "F", "H", "I",  "L",
                   "M", "N", "P", "R", "S",  "U", "W", "Y",
                   "Aw", "Bk", "Cm", "Ik", "Om", "Um"]

    def __int__(self, model_path, handlandmarks):
        self.model_path = model_path
        self.model = self.load_model(model_path)
        self.handlandmarks = handlandmarks

    # read model
    def load_model(self, model_path):
        model = pickle.load(open(model_path, 'rb'))
        return model

    # get landmarks
    def get_landmarks(self, handlandmarks):
        return True

    # get left or right hand
    def get_L_or_R(self, handlandmarks):
        return True

    # transform landmarks if left
    def mirror_landmarks(self, handlandmarks):
        return True

    # normalize landmarks
    def get_normalized_landmarks(self, hand_landmarks):
        X = [hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x,
             hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x,
             hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x,
             hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x,
             hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x,
             hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x,
             hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x,
             hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x,
             hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
             hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x,
             hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x,
             hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x,
             hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x,
             hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x,
             hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x,
             hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x,
             hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x,
             hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x,
             hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x,
             hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x,
             hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x]

        Y = [hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y,
             hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y,
             hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y,
             hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y,
             hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y,
             hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y,
             hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y,
             hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y,
             hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y,
             hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y,
             hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y,
             hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y,
             hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y,
             hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y,
             hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y,
             hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y,
             hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y,
             hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y,
             hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y,
             hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y,
             hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y]

        x0 = min(X)
        y0 = min(Y)
        x1 = max(X)
        y1 = max(Y)

        X_new = np.zeros(21)
        Y_new = np.zeros(21)

        # przesunięcie do nowego 0
        for i in range(0, 20):
            X_new[i] = (X[i] - x0)
            Y_new[i] = (Y[i] - y0)

        return X_new, Y_new

    # get new line for classification
    def get_line(self, handlandmarks):
        X, Y = self.get_normalized_landmarks(handlandmarks)
        line =  np.array([[X[0], Y[0], X[1], Y[1], X[2], Y[2], X[3], Y[3],
            X[4], Y[4], X[5], Y[5], X[6], Y[6], X[7], Y[7], X[8], Y[8],
            X[9], Y[9], X[10], Y[10], X[11], Y[11], X[12], Y[12], X[13], Y[13],
            X[14], Y[14], X[15], Y[15], X[16], Y[16], X[17], Y[17], X[18], Y[18],
            X[19], Y[19], X[20], Y[20] ]])
        return line


    # get result from classifier
    def get_result(self, model, handlandmarks):
        line = self.get_line(handlandmarks)
        result = model.predict(line)
        return result

    def result_parser(self, result):
        i = self.class_nums.index(result)
        return self.class_names[i]


class TestHandClassifierHandler:

    def test_result_parser(self):
        # czy udało się wyciągnąć punkty chakakterystyczne z funckji
        hch = HandClassifierHandler()
        result = 0
        result1 = 28

        assert hch.result_parser(result) == "0"
        assert hch.result_parser(result1) == "Um"








