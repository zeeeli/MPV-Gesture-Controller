import cv2
import mediapipe as mp
import time
import numpy as np


class handDetector:
    def __init__(self, mode=False, max_hands=2, complexity=1, confidence_detect=0.5, confidence_track=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.complexity = complexity
        self.confidence_detect = confidence_detect
        self.confidence_track = confidence_track

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, self.complexity,
                                         self.confidence_detect, self.confidence_track)
        self.results = None

    def detect_hands(self, img, draw=True, bbox=True, mirror=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        hands_data = []  # Two index long list of hand dictionary
        if self.results.multi_hand_landmarks:
            for landmarks, handedness in zip(self.results.multi_hand_landmarks, self.results.multi_handedness):
                hand_dict = {}  # Dictionary to store value of hands
                idx = []  # List for landmark index
                position = []  # List for x positions

                # Handedness controller
                if mirror:
                    if handedness.classification[0].label == "Right":
                        hand_dict["Hand"] = "Left"
                    else:
                        hand_dict["Hand"] = "Right"
                else:
                    hand_dict["Hand"] = handedness.classification[0].label

                # Extract hand positions
                for i, lm in enumerate(landmarks.landmark):
                    # x, y, z = int(lm.x * w), int(lm.y * h), int(lm.z * c)
                    idx.append(i)
                    position.append([lm.x, lm.y, lm.z])
                # Append lists to dictionary
                hand_dict["lm"] = idx
                hand_dict["position"] = position

                # Append dictionary to master list
                hands_data.append(hand_dict)

                # Creating a bounded box
                bound = self.__get_bound(img, landmarks)
                # left_side, right_side, top_side, bottom_side = min(pos_x), max(pos_x), max(pos_y), min(pos_y)

                # Drawing Landmarks and bounding box
                if draw:
                    self.mp_draw.draw_landmarks(
                        img,
                        landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=4, circle_radius=3),
                        self.mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2))

                if bbox:
                    cv2.rectangle(img, (bound[0]-8, bound[1]), (bound[2]+8, bound[3]), (0, 0, 235), 1)
                    cv2.putText(img, hand_dict["Hand"], (bound[0] + 5, bound[1] - 4), cv2.FONT_HERSHEY_SIMPLEX,
                                1.5, (0, 0, 235), 2, cv2.LINE_AA)

        if draw:
            return hands_data, img
        else:
            return hands_data

    @staticmethod
    def split_hands(hands_list):
        left = {}
        right = {}

        # Split lists into Left and Right hand dictionaries
        for i, d in enumerate(hands_list):
            if d["Hand"] == "Left":
                left = hands_list[i]
            elif d["Hand"] == "Right":
                right = hands_list[i]
            else:
                continue

        return left, right

    @staticmethod
    def __get_bound(img, landmarks):
        w, h = img.shape[1], img.shape[0]
        bound_array = np.empty((0, 2), int)

        for i, landmark in enumerate(landmarks.landmark):
            x = min(int(landmark.x * w), w - 1)
            y = min(int(landmark.y * h), h - 1)
            point = [np.array((x, y))]
            bound_array = np.append(bound_array, point, axis=0)

        x, y, w, h = cv2.boundingRect(bound_array)

        return [x, y, x+w, y+h]

    @staticmethod
    def flat_positions(left_hand_dict, right_hand_dict):
        if len(left_hand_dict) == 0:
            lh = np.zeros(21 * 3)
        else:
            lh = left_hand_dict.pop("position")
            lh = np.array(lh).flatten()

        if len(right_hand_dict) == 0:
            rh = np.zeros(21 * 3)
        else:
            rh = right_hand_dict.pop("position")
            rh = np.array(rh).flatten()

        return lh, rh


def test():
    cap = cv2.VideoCapture(0)
    cap.set(3, 620)
    cap.set(4, 480)
    t0 = 0
    t1 = 0
    detector = handDetector()
    while cap.isOpened():
        # Capturing image from camera
        success, img = cap.read()
        if not success:
            print("Empty frame ignored")
            continue

        # Display hand connection
        hands, img = detector.detect_hands(img)

        # Split hand data into left and right hand
        left_hand_dict, right_hand_dict = detector.split_hands(hands)

        # Convert positions to flat numpy arrays
        # create zero arrays if empty for tensorflow training
        lh_position, rh_position = detector.flat_positions(left_hand_dict, right_hand_dict)


        print(len(hands))
        print(rh_position)
        print(lh_position)

        # Display fps on camera stream
        t1 = time.time()
        fps = 1/(t1 - t0)
        t0 = time.time()
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        cv2.imshow("Hand Detection", img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test()