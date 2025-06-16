import cv2
import threading
import time
class VideoStreamThread:
    def __init__(self, src, retry_interval=5):
        self.src = src
        self.retry_interval = retry_interval
        self.capture = None
        self.ret = False
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _connect(self):
        print(f"[INFO] Attempting to connect to stream: {self.src}")
        cap = cv2.VideoCapture(self.src)
        if cap.isOpened():
            print("[INFO] Stream connected successfully.")
            return cap
        else:
            print("[WARNING] Stream connection failed.")
            return None

    def _update(self):
        while self.running:
            if self.capture is None or not self.capture.isOpened():
                self.capture = self._connect()
                if self.capture is None:
                    time.sleep(self.retry_interval)
                    continue

            ret, frame = self.capture.read()
            if not ret:
                print("[WARNING] Frame grab failed. Reconnecting...")
                self.capture.release()
                self.capture = None
                time.sleep(self.retry_interval)
                continue

            with self.lock:
                self.ret = ret
                self.frame = frame
            # time.sleep(0.02)    

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.ret and self.frame is not None else (False, None)

    def stop(self):
        self.running = False
        self.thread.join()
        if self.capture:
            self.capture.release()

