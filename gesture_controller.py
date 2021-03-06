import cv2
import numpy as np
import os
from keras.callbacks import TensorBoard
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from matplotlib import pyplot as plt
from scipy import stats
from hand_detection import handDetector
import time


class gestureController:
    def __init__(self):
        self.commands = np.array(open("gestures.txt").read().splitlines())
        self.model = Sequential()
        self.model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 126))),
        self.model.add(LSTM(128, return_sequences=True, activation='relu')),
        self.model.add(LSTM(64, return_sequences=False, activation='relu')),
        self.model.add(Dense(64, activation='relu')),
        self.model.add(Dense(32, activation='relu')),
        self.model.add(Dense(self.commands.shape[0], activation='softmax')),
        self.model.load_weights('gestures.h5')
        self.target_string = None
        self.gesture_name = []
        self.predictions = []

    def get_gesture(self, img, keypoints, hands, visualize=True, min_accuracy=0.9):
        threshold = min_accuracy
        if len(keypoints) == 30 and len(hands) == 1:
            result = self.model.predict(np.expand_dims(keypoints, axis=0))[0]
            self.predictions.append(np.argmax(result))

            if np.unique(self.predictions[-10:])[0] == np.argmax(result):
                if result[np.argmax(result)] > threshold:
                    if len(self.gesture_name) > 0:
                        if self.commands[np.argmax(result)] != self.gesture_name[-1]:
                            self.gesture_name.append(self.commands[np.argmax(result)])
                            # self.target_string = self.gesture_name[-1]
                    else:
                        self.gesture_name.append(self.commands[np.argmax(result)])

                    self.target_string = self.gesture_name[-1]

            if len(self.gesture_name) > 5:
                self.gesture_name = self.gesture_name[-5:]
            if visualize:
                # Visualizing the probability of the gestures
                img = self.__draw_probability(img, result, self.commands)

        return img, self.target_string, self.gesture_name

    @staticmethod
    def __draw_probability(img, result, gesture):
        output_img = img.copy()
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
        for i, probability in enumerate(result):
            cv2.rectangle(output_img, (0, 60 + i * 40), (int(probability * 100), 90 + i * 40), colors[i], -1)
            cv2.putText(output_img, gesture[i], (0, 85 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)
        return output_img


def test():
    cap = cv2.VideoCapture(0)
    cap.set(3, 620)
    cap.set(4, 480)
    t0 = 0
    t1 = 0
    detector = handDetector()
    controller = gestureController()
    keypoints = []
    while cap.isOpened():
        # Collect image from camera
        success, img = cap.read()

        # Make hand detection, split hand data
        hands, img = detector.detect_hands(img, bbox=False)
        lh, rh = detector.split_hands(hands)

        # Gathering keypoint data of left and right hands
        lh_pos, rh_pos = detector.flat_positions(lh, rh)
        kp = np.concatenate([lh_pos, rh_pos])
        keypoints.append(kp)
        keypoints = keypoints[-30:]

        # Getting gesture information and drawing
        img, gesture, gesture_list = controller.get_gesture(img, keypoints, hands, visualize=True, min_accuracy=0.9)
        print(gesture)
        cv2.rectangle(img, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(img, ' '.join(gesture_list), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display fps on camera stream
        t1 = time.time()
        fps = 1/(t1 - t0)
        t0 = time.time()
        cv2.putText(img, str(int(fps)), (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Hand Detection", img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break



if __name__ == '__main__':
    test()
