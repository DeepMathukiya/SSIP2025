import cv2
import threading
from datetime import datetime
import os
import time

class DictionaryThread:
    def __init__(self, SAVE_FRAMES=False,recognized_folder=None, unrecognized_folder=None):
        self.data =  {}
        # self.retry_interval = retry_interval
        self.lock = threading.Lock()
        self.running = True
        self.SAVE_FRAMES = SAVE_FRAMES  # Assuming you want to save frames by default
        self.thread = threading.Thread(target=self._save, daemon=True)
        self.thread.start()
        self.recognized_folder = recognized_folder
        self.unrecognized_folder = unrecognized_folder

    def _update(self , track_id=None, label=None, frame=None, max_sim=0.0):
        with self.lock:
            if track_id not in self.data.keys():
                self.data[track_id] = {
                    'label': label,
                    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                    'counter': 1,
                    'frame': frame.copy(),
                    'time': time.time(),
                    'similarity': max_sim
                    }   
            else: 
                if self.data[track_id]['similarity']< max_sim:
                    self.data[track_id]['label'] = label
                    self.data[track_id]['timestamp'] = datetime.now().strftime("%Y%m%d_%H%M%S")
                    self.data[track_id]['frame'] = frame.copy()
                    self.data[track_id]['similarity'] = max_sim   
                
                self.data[track_id]['time'] = time.time()  
                self.data[track_id]['counter'] += 1
                # self.data.update(dictionary)
    def _save(self):
        if self.SAVE_FRAMES:
            while self.running:
                if self.data:
                    with self.lock:
                        k = self.data.copy()  # Create a copy of the data to avoid issues while iterating
                        for id, item in k.items():
                            if  time.time() - item['time'] > 10:
                                dict_data = self.data.pop(id) 
                                self.save_img(id ,dict_data)
                                print(f"[INFO] Removed ID {id} from dictionary after 10 seconds of inactivity.")
                time.sleep(5)  # Sleep for a while before checking again                   
                                
    def save_img(self , id ,dict_data):
        if dict_data:
            file_name = f"{dict_data['label']}_{dict_data['timestamp']}_{dict_data['counter']}.jpg"   
            if (dict_data['label']== 'Unknown'):
                file_path = os.path.join(self.unrecognized_folder, file_name)
            else:
                file_path=os.path.join(self.recognized_folder ,file_name)    
            cv2.imwrite(file_path, dict_data['frame'])   
            print(f"[INFO] Saved frame for ID {id} as{file_path} / {file_name}")
            
    def stop(self):
        with self.lock:
            for id, item in self.data.items():
                self.save_img(id ,item)
        self.running = False
        self.thread.join()