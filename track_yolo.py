# Version 1.0 - adding tracking using SORT
# pre: register_known_faces_3.py : only detection and recognition


# Version 3.0 - RTMP Stream Face Recognition with Timestamped Output (Modified for recognition-based saving) with fps and threading and SORT tracking

import os
import cv2
import argparse
import numpy as np
from datetime import datetime
import time
from collections import deque
from sklearn.metrics.pairwise import cosine_similarity
# from insightface.app import FaceAnalysis
from camera import VideoStreamThread  # Import the threaded grabber
# from sort import Sort  # Import SORT tracker
from face_rec_dict import DictionaryThread  # Import the dictionary thread for saving frames
from ultralytics import YOLO
from collections import defaultdict

# ========== Argument Parsing ==========
parser = argparse.ArgumentParser(description="Face recognition on RTMP stream using InsightFace")
parser.add_argument("--rtmp", required=True, help="RTMP stream URL (e.g., rtmp://...)")
parser.add_argument("--save", action="store_true", help="Save snapshots with detections")
args = parser.parse_args()

RTMP_URL = args.rtmp
SAVE_FRAMES = args.save

# ========== CONFIG ==========
EMBEDDINGS_FOLDER = "embeddings"
OUTPUT_FOLDER = "output_results"
SIMILARITY_THRESHOLD = 0.35
FONT = cv2.FONT_HERSHEY_SIMPLEX
fps_buffer = deque(maxlen=30)

# ========== Output Directories ==========
if SAVE_FRAMES:
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_folder = os.path.join(OUTPUT_FOLDER, f"{run_timestamp}_stream_output")
    recognized_folder = os.path.join(base_output_folder, "recognized")
    unrecognized_folder = os.path.join(base_output_folder, "unrecognized")
    os.makedirs(recognized_folder, exist_ok=True)
    os.makedirs(unrecognized_folder, exist_ok=True)

# ========== Load Embeddings ==========
# known_embeddings = []
# known_names = []

# for file in os.listdir(EMBEDDINGS_FOLDER):
#     if file.endswith(".npy"):
#         name = os.path.splitext(file)[0]
#         emb = np.load(os.path.join(EMBEDDINGS_FOLDER, file))
#         emb = emb / np.linalg.norm(emb)
#         known_embeddings.append(emb)
#         known_names.append(name)

# embedding_matrix = np.vstack(known_embeddings)

# ========== Init Face Detector ==========
# face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# face_app.prepare(ctx_id=0)
yolo = YOLO('yolo_model_trined_on_head.pt')

# ========== Init Tracker ==========
# tracker = Sort(max_age=3, min_hits=2, iou_threshold=0.3)

# ========== Start Video Stream ==========
cap = VideoStreamThread(RTMP_URL)

# ========== Initialize Dictionary Thread ==========
dict_thread = DictionaryThread(SAVE_FRAMES=SAVE_FRAMES , recognized_folder=recognized_folder, unrecognized_folder=unrecognized_folder)

print(f"[INFO] Streaming from: {RTMP_URL} (press 'q' to quit)")
frame_id = 0
track_history = defaultdict(lambda: [])
# ========== Main Loop ==========
while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret or frame is None:
        continue
    result = yolo.track(frame,device="cuda", persist=True)[0]

        # Get the boxes and track IDs
    if result.boxes and result.boxes.is_track:
        boxes = result.boxes.xywh.cpu()
        track_ids = result.boxes.id.int().cpu().tolist()
        scores = result.boxes.conf.cpu().tolist()
        # Visualize the result on the frame
        frame = result.plot()
        # print(result)
        # Plot the tracks
        for box, track_id,score in zip(boxes, track_ids , scores):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 30 tracks for 30 frames
                track.pop(0)
            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            print(f"Track ID: {track_id}, Score: {score:.2f}")
            cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
            dict_thread._update(
            track_id=track_id,
            label= f"ID_{track_id}",
            frame=frame.copy(),
            max_sim=score
            )
        
    
        
    # FPS Calculation
    elapsed_time2 = time.time() - start_time
    time.sleep(abs(0.1-elapsed_time2))
    elapsed_time = time.time() - start_time

    fps_buffer.append(1 / elapsed_time if elapsed_time > 0 else 0)
    smoothed_fps = sum(fps_buffer) / len(fps_buffer)
    fps_ = 1/elapsed_time
    cv2.putText(frame, f"FPS: {fps_:.2f}", (10, 30), FONT, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Smoothed FPS: {smoothed_fps:.2f}", (10, 60), FONT, 0.8, (0, 255, 255), 2)
    # Show Frame
    cv2.imshow("RTMP Face Recognition", frame)
      # Small delay to allow GUI to update
    # # Save frame if enabled
    # if SAVE_FRAMES and faces:
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     filename = f"frame_{timestamp}.jpg"
    #     folder = recognized_folder if recognized_anyone else unrecognized_folder
    #     cv2.imwrite(os.path.join(folder, filename), frame)
    #     print(f"[INFO] Saved frame to {os.path.join(folder, filename)}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.stop()
cv2.destroyAllWindows()
dict_thread.stop()  # Stop the dictionary thread gracefully
print("[INFO] Stream ended and resources released.")
