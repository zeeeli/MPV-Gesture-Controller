import cv2
import numpy as np
import os
from hand_detection import handDetector

# Number of collections needed to train
NUMBER_CAPTURES = 30

# Length of each data collection period (in frames)
CAPTURE_LENGTH = 30

# Path for data
PATH = os.path.join('Training_Data')

# Commands that will be trained
commands = np.array(open("gestures.txt").read().splitlines())

# Creating suborders for training data storage
for c in commands:
    for i in range(NUMBER_CAPTURES):
        try:
            os.makedirs(os.path.join(PATH, c, str(i)))
        except:
            pass

# Training Loop
cap = cv2.VideoCapture(0)
cap.set(3, 620)
cap.set(4, 480)
detector = handDetector()

# Looping through gesture command names
for command in commands:
    # Looping through number of captures
    for capture in range(NUMBER_CAPTURES):
        # Loop through all individual frames of one capture
        for frame in range(CAPTURE_LENGTH):
            # Collect image from camera
            success, img = cap.read()

            # Make hand detection, split hand data
            hands, img = detector.detect_hands(img, bbox=False)
            lh, rh = detector.split_hands(hands)

            # Drawing collection environment on capture
            if frame == 0:
                cv2.putText(img, 'STARTING COLLECTION', (80, 200), cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (255, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(img, 'Capture #{} For {} '.format(capture, command), (15, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('Feed', img)
                cv2.waitKey(1500)
            else:
                cv2.putText(img, 'Capture #{} For {} '.format(capture, command), (15, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('Feed', img)

            # Extracting hand keypoints and saving in file system
            lh_pos, rh_pos = detector.flat_positions(lh, rh)
            kp = np.concatenate([lh_pos, rh_pos])

            # #Uncomment to Overwrite current training data
            # cap_path = os.path.join(PATH, command, str(capture), str(frame))
            # np.save(cap_path, kp)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()








