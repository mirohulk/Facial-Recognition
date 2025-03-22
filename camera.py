import cv2
from threading import Thread

class Camera:
    def __init__(self, src=0, width=1280, height=720):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.ret, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.stream.read()

    def read(self):
        return self.frame.copy()

    def stop(self):
        self.stopped = True
        self.stream.release()
