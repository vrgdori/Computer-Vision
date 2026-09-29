import cv2
import threading
import numpy as np
from pathlib import Path
import time
import config

class AsyncVideoStream:
    def __init__(self, src=None, input_type=None):
        self.started = False
        self.read_lock = threading.Lock()
        self.image_mode = False
        self.image_paths = []
        self.image_index = 0
        self.cap = None

        self.input_type = input_type or config.INPUT_TYPE
        self.src = src if src is not None else config.INPUT_SOURCE

        if self.input_type == "FOLDER":
            path_obj = Path(self.src)
            if path_obj.exists() and path_obj.is_dir():
                self.image_paths = sorted(
                    list(path_obj.rglob('*.jpg')) + 
                    list(path_obj.rglob('*.png')) + 
                    list(path_obj.rglob('*.jpeg')) +
                    list(path_obj.rglob('*.webp'))
                )
            
            if self.image_paths:
                self.image_mode = True
                self.grabbed = True
                self.frame = cv2.imread(str(self.image_paths[0]))
                print(f"[AsyncVideoStream] {len(self.image_paths)} kép betöltve innen: {self.src}")
            else:
                print(f"[AsyncVideoStream] HIBA: A megadott mappa nem tartalmaz képeket: {self.src}")
                self.grabbed = False
                self.frame = None

        else: # "RTSP" vagy "CAMERA"
            print(f"[AsyncVideoStream] Csatlakozás a streamhez: {self.src}")
            if isinstance(self.src, int) or (isinstance(self.src, str) and self.src.isdigit()):
                self.cap = cv2.VideoCapture(int(self.src), cv2.CAP_DSHOW) # Windows DirectShow
            else:
                self.cap = cv2.VideoCapture(str(self.src))

            self.grabbed, self.frame = self.cap.read()
            if not self.grabbed:
                print("[AsyncVideoStream] HIBA: Videófolyam nem érhető el.")

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()
        return self
    
    def update(self):
        while self.started:
            if self.image_mode:
                time.sleep(0.05) # ~20 FPS léptetés a képek között
                if self.image_paths:
                    self.image_index = (self.image_index + 1) % len(self.image_paths)
                    img_path = self.image_paths[self.image_index]
                    frame = cv2.imread(str(img_path))
                    
                    with self.read_lock:
                        if frame is not None:
                            self.frame = frame
                            self.grabbed = True
            else:
                if self.cap and self.cap.isOpened():
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
        if self.cap and self.cap.isOpened():
            self.cap.release()