import cv2 #pip install opencv-python
import threading

class AsyncVideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            if not grabbed:
                self.stop()
                break
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            if not self.grabbed or self.frame is None:
                return None
            return self.frame.copy()

    def stop(self):
        self.started = False
        if self.cap.isOpened():
            self.cap.release()