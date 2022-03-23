import cv2
import numpy as np
import os

from keras.callbacks import TensorBoard
from keras.layers import LSTM, Dense

from hand_detection import handDetector
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential


def trainer():
    logging_path = os.path.join('logs')
    tensor_callback = TensorBoard(log_dir=logging_path)
    print(model.summary())
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit(X_train, Y_train, epochs=200, callbacks=[tensor_callback])
    model.save('gestures.h5')


# Number of collections needed to train
NUMBER_CAPTURES = 30

# Length of each data collection period (in frames)
CAPTURE_LENGTH = 30

# Path for data
PATH = os.path.join('Training_Data')

# gesture that will be trained
commands = np.array(open("gestures.txt").read().splitlines())

# Put name of gestures in dict
label_dict = {l: n for n, l in enumerate(commands)}

# Pre processing data
training_data, gesture_labels = [], []
for command in commands:
    for capture in range(NUMBER_CAPTURES):
        capture_frames = []
        for frame in range(CAPTURE_LENGTH):
            frame_data = np.load(os.path.join(PATH, command, str(capture), "{}.npy".format(frame)))
            capture_frames.append(frame_data)
        training_data.append(capture_frames)
        gesture_labels.append(label_dict[command])

# Data of landmarks from data collection
X = np.array(training_data)

# Binary flags representing gesture name
Y = to_categorical(gesture_labels).astype(int)

# Splitting arrays for use in training
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05)

# ###############################################
# # Building LSTM Neural Network
# ###############################################
model = Sequential()
# model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=X_train[0].shape))
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 126)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(commands.shape[0], activation='softmax'))
# trainer()
model.load_weights('gestures.h5')

# Evaluating model
Y_hat = model.predict(X_train)
Y_true = np.argmax(Y_train, axis=1).tolist()
Y_hat = np.argmax(Y_hat, axis=1).tolist()

print(multilabel_confusion_matrix(Y_true, Y_hat))
print(accuracy_score(Y_true, Y_hat))
