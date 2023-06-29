import cv2
from ultralytics import YOLO
import numpy as np
import shutil
import os
from pathlib import Path

def add_frame_number(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = 720
    height = 480
    out = cv2.VideoWriter(output_path, codec, fps, (width, height))
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        out.write(frame)
        print(f'done {frame_count} frames')

        if frame_count == 100:
            break
    cap.release()
    out.release()


def get_frame_from_vid(
    vid_fp,
    ls_range,
    save_dir
):
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(vid_fp)
    cnt = 0

    all_indices = []
    for start_idx, end_idx in ls_range:
        all_indices += list(range(start_idx, end_idx+1))
    max_idx = max(all_indices)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cnt += 1
        if cnt > max_idx:
            break

        for start_fr_idx, end_fr_idx in ls_range:
            if start_fr_idx <= cnt <= end_fr_idx:
                cv2.imwrite(os.path.join(save_dir, f'frame_{cnt}.jpg'), frame)
                print(f'saved frame {cnt}')
                break


def draw_circle_on_img():
    img = cv2.imread('debug/test_4/frame_4003_4023/frame_30188.jpg')
    coord = [1616, 921]
    cv2.circle(img, coord, 10, (0, 0, 255), 2)
    cv2.imwrite('a.jpg', img)



if __name__ == '__main__':
    draw_circle_on_img()
    # ls_range = [
    #     (4003, 4023),
    #     (12295, 12314),
    #     (15664, 15684),
    #     (18937, 18957),
    #     (18974, 18994),
    #     (22234, 22254),
    #     (30679, 30699),
    #     (31681, 31701),
    #     (28698, 28718),
    # ]
    # get_frame_from_vid(
    #     vid_fp='samples/test_4.mp4',
    #     ls_range = ls_range,
    #     save_dir=f'debug/test_4/frame_4003_4023'
    # )
