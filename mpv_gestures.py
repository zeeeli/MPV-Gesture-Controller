import cv2
import numpy as np
import time
import mpv
import sys
from hand_detection import handDetector
from gesture_controller import gestureController

# Initialize MPV player
quality = 'bestvideo[height<=?720][fps<=?30][vcodec!=?vp9]+bestaudio/best'
# player = mpv.MPV(log_handler=print, loglevel='debug', ytdl=True,  ytdl_format=quality)
player = mpv.MPV(ytdl=True, ytdl_format=quality, osc=True)


def main():
    try:
        player.play(str(sys.argv[1]))
    except IndexError:
        print('ERROR: No file or link specified')
        print('Filenames must include file type')
        sys.exit(1)

    # Initialize Webcam Capture
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
        img, gesture, gesture_list = controller.get_gesture(img, keypoints, hands, visualize=False, min_accuracy=0.9)
        cv2.rectangle(img, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(img, ' '.join(gesture_list), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Translate gesture to MPV keybinding
        get_commands(gesture)

        # Display fps on camera stream
        t1 = time.time()
        fps = 1 / (t1 - t0)
        t0 = time.time()
        cv2.putText(img, str(int(fps)), (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Hand Detection", img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def get_commands(gesture):
    if gesture == 'Pause':
        player.pause = True
    if gesture == 'Play':
        if player.pause is True:
            player.pause = False
        else:
            pass
    if gesture == 'Vol Up':
        player.pause = False
        if player.volume <= 125:
            player.volume += 5
        else:
            pass
    if gesture == 'Vol Down':
        player.pause = False
        if player.volume >= 5:
            player.volume -= 5
        else:
            pass
    if gesture == 'Quit':
        player.stop()


if __name__ == '__main__':
    main()
