from threading import Thread
import cv2

import logging
logger = logging.getLogger(__name__)


class VideoStream:
    def __init__(self, src=0, is_file=False):
        self.src = src
        self.stream = self.connect_to_camera()

        self.grabbed, self.frame = self.stream.read()
        while not self.grabbed:
            if not is_file:
                logger.warning(f"Camera status is False, reconnecting to camera")
                self.stream = self.connect_to_camera()
                self.grabbed, self.frame = self.stream.read()
            else:
                break

        self.stopped = False
        self.record = False
        self.out = None
        self.name = None
        self.is_file = is_file

    def connect_to_camera(self, gstreamer=False):
        if gstreamer:
            stream = cv2.VideoCapture(self.src, cv2.CAP_GSTREAMER)
        else:
            stream = cv2.VideoCapture(self.src)
        return stream

    def start(self):
        if not self.is_file:
            Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        camera_status = False
        while True:
            if self.stopped:
                break

            self.grabbed, self.frame = self.stream.read()

            if not self.grabbed and camera_status:
                camera_status = False

            while not self.grabbed:
                logger.warning(f"Camera status is False, reconnecting to camera")
                self.stream = self.connect_to_camera()
                if self.stream.isOpened():
                    self.grabbed, self.frame = self.stream.read()
                else:
                    self.grabbed =False
                    logger.warning(f"Camera stream is not opened")
                    break

            if self.grabbed and not camera_status:
                camera_status = True
        self.stream.release()

    def read(self):
        if self.is_file:
            self.grabbed, self.frame = self.stream.read()
            return self.grabbed, self.frame
        else:
            return self.grabbed, self.frame

    def release(self):
        self.stopped = True
