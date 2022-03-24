### *Project Intended For My Personal Learning (Not Maintained)*

# MPV-Gesture-Controller

Use hand gestures as MPV keybinds.

![Hand gestures used](/images/gestures_BnW.png)


## Files Explained
`hand_detection.py` - Detects hands and stores hand position and handedness in dictionary

`gesture_data_collector.py` - Logs user inputted sample of gestures as flattened numpy arrays. 30 frames for each of the 30 collection samples for each gesture

`gesture_train.py` - Creates LSTM Reccurent Neural Network using collected gesture data from logs and saves model as `gestures.h5`

`gesture_controller.py` - Contains logic that compares user input in webcam to Neural Network and returns recognized gesture

`mpv_gestures.py` - Interfaces with MPV video player using recognized gestures inputted from camera

## Dependencies
### Libraries
- `MediaPipe`
- `TensorFlow`
- `OpenCV`
- `numpy`
- [`python-mpv`](https://github.com/jaseg/python-mpv/)

### Linux Packages
- `libmpv-dev`
- [`mpv`](https://github.com/mpv-player/mpv/) (Youtube streams work best with >0.32 releases)

## Usage
Run `mpv_gestures.py` with either a path to a video file OR a Youtube link as an argument




