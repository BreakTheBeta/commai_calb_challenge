from typing import Tuple
from pathlib import Path
import torch
import torchvision
import matplotlib.pyplot as plt
import cv2
import numpy as np
import plac


FPS = 20
NUM_VIDEOS = 5

def load_data(video_id) -> Tuple[cv2.VideoCapture, np.ndarray, np.ndarray]:
    video_path = f"calib_challenge/labeled/{video_id}.hevc"
    label_path = f"calib_challenge/labeled/{video_id}.txt"

    vc = cv2.VideoCapture(video_path)
    labels_r = np.genfromtxt(label_path, delimiter=" ")
    labels_d = np.rad2deg(labels_r)

    return vc, labels_r, labels_d

def main(video_id: int):
    video_id = int(video_id)
    assert video_id < NUM_VIDEOS

    cap, labels_r, labels_d = load_data(video_id)
    assert cap.isOpened()

    winname = "Frame"
    cv2.namedWindow(winname)
    cv2.setWindowProperty(winname, cv2.WND_PROP_TOPMOST, 1)
    
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pitch_d, yaw_d = labels_d[i]
        cv2.putText(frame, f"P{pitch_d:.2f} Y{yaw_d:.2f} {i}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0, 0, 0), thickness=8, lineType=1)
        cv2.putText(frame, f"P{pitch_d:.2f} Y{yaw_d:.2f} {i}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(20, 150, 150), thickness=2, lineType=1)
        cv2.imshow(winname, frame)
        if cv2.waitKey(1000//FPS) & 0xFF == ord('q'):
            break
        i += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    plac.call(main)