import logging
# Device settings
DEVICE = 'VELANKANI'
# DEVICE = 'test'
# DEVICE_ID = "VELANKANI"
DEVICE_ID = 1

DELETE_DAYS = 5

# Stream setings
STREAMS = 1
LANES = [1,2]
# STREAM_LINKS = ["rtsp://service:Ncrtc!123$@169.169.69.10/?inst=2"]
# STREAM_LINKS = ["rtsp://admin:admin@169.169.69.103:554/1/h265major/"]
# STREAM_LINKS = ["rtsp://admin:admin123@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0"]
STREAM_LINKS = ["rtsp://admin:Admin123@10.1.68.50:554"]
# STREAM_LINKS=['/home/heet/Downloads/vlc-record-2023-04-04-12h23m06s-rtsp___192.168.0.100_554_-.avi']
# STREAM_LINKS = ["rtsp://admin:admin123@192.168.0.100:554/cam/realmonitor?channel=1&subtype=0"]
# STREAM_LINKS=[0]

ROI = [237, 100, 980, 600]

# Model settings
FACE_DETECTION_MODEL = 'models/frozen_models/yolov5n-face-320.onnx'
FACE_RECOGNIZER_MODEL = 'models/frozen_models/ms1mv3_r34.onnx'

# FACE_RECOGNIZER_MODEL = ''
MIN_FACE_AREA = 0
FACE_RECOGNIZER_THRESH = 0.9
FACE_RECOGNIZER_THRESH_VISITOR = 0.5

PERSON_DETECTION_MODEL = './models/frozen_models/yolov5s.onnx'
PERSON_CONF = 0.7

# Program settings
MAX_DISAPPEARED = 2  # For centroid tracking

USER_DATA_PATH = 'data/users_data'

# Debug Settings
SHOW_VIDEO = True
RECORD_VIDEO = True
