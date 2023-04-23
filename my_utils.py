import os
from pathlib import Path
import cv2
from PIL import Image
import pdb
from ultralytics import YOLO
from ultralytics.yolo.engine.results import Results
import numpy as np
from collections import Counter
from scipy.signal import savgol_filter, find_peaks
from predictor import *

def compute_area(row: np.array):
    xmin, ymin, xmax, ymax = row.tolist()
    w = xmax - xmin
    h = ymax - ymin
    return w * h


def get_game_info(
    ls_ball_center_x,
    ls_ball_center_y,
    ls_table_contour: list,
    ls_person_bb,
    calibre_table_position=False,
    smooth=False,
    distance_x=50,
    prominence_x=500,
    distance_y=10,
    prominence_y=10,
):
    game_info = {}
    
    ls_table_bb = []
    for contour in ls_table_contour:
        xmin = np.min(contour[:, 0])
        xmax = np.max(contour[:, 0])
        ymin = np.min(contour[:, 1])
        ymax = np.max(contour[:, 1])
        ls_table_bb.append([xmin, ymin, xmax, ymax])

    if calibre_table_position:  # assume that table is fixed during the game
        # find most frequent table position using Counter
        most_freq_table_bb = Counter(ls_table_bb).most_common(1)[0][0]
        ls_table_contour = [most_freq_table_bb] * len(ls_table_bb)
        # pdb.set_trace()

    ls_valid_x = [el for el in ls_ball_center_x if el != -1]
    ls_valid_x_idx = [i for i, el in enumerate(ls_ball_center_x) if el != -1]
    ls_valid_y = [el for el in ls_ball_center_y if el != -1]
    ls_valid_y_idx = [i for i, el in enumerate(ls_ball_center_y) if el != -1]

    if smooth:
        ls_valid_x = savgol_filter(ls_valid_x, 31, 3)
        ls_valid_y = savgol_filter(ls_valid_y, 11, 1)

    # Find the local maxima using the find_peaks function
    # distance between 2 peaks is 10 frame, distance between 2 consecutive extrema is 500
    ls_maximum_idx, _ = find_peaks(ls_valid_x, distance=distance_x, prominence=prominence_x)
    ls_minimum_idx, _ = find_peaks(-np.array(ls_valid_x), distance=distance_x, prominence=prominence_x)
    ls_maximum_idx = [ls_valid_x_idx[el] for el in ls_maximum_idx.tolist()]
    ls_minimum_idx = [ls_valid_x_idx[el] for el in ls_minimum_idx.tolist()]

    ls_bounce_idx, _ = find_peaks(ls_valid_y, distance=distance_y, prominence=prominence_y, width=5)
    ls_bounce_idx = [ls_valid_y_idx[el] for el in ls_bounce_idx.tolist()]
    for idx in ls_bounce_idx:
        game_info[idx] = 'bounce'

    extrema = sorted(ls_maximum_idx + ls_minimum_idx)
    extrema = [0] + extrema + [len(ls_ball_center_x)-1]
    nbounce2type = {
        0: 'out',
        1: 'valid',
        2: 'serve',
        3: 'invalid'
    }
    for i in range(len(extrema)-1):
        start = extrema[i]
        end = extrema[i+1]
        # check how many idx in ls_bounce_idx are in between start and end
        ls_bounce_idx_in_between = [el for el in ls_bounce_idx if start < el < end]
        num_bounce = len(ls_bounce_idx_in_between)
        num_bounce = min(3, num_bounce)
        game_info[(start, end)] = nbounce2type[num_bounce]

    return game_info


def get_ls_ball_center(processed_results, frame_limit=1e9, save=False, out_fp=None):
    if save and not Path(out_fp).parent.exists():
        os.makedirs(Path(out_fp).parent)
    cnt = 0
    ls_ball_center_x, ls_ball_center_y = [], []
    for orig_img, table_contour, ball_bb, person_bbs in processed_results:
        if ball_bb is not None:
            ls_ball_center_x.append(int((ball_bb[0] + ball_bb[2])//2))
            ls_ball_center_y.append(int((ball_bb[1] + ball_bb[3])//2))
        else:
            ls_ball_center_x.append(-1)
            ls_ball_center_y.append(-1)
        cnt += 1
        if cnt == frame_limit:
            break
    if save:
        out_fp_x = out_fp.replace('.txt', '_x.txt')    
        out_fp_y = out_fp.replace('.txt', '_y.txt')
        with open(out_fp_x, 'w') as f:
            f.write(' '.join([str(el) for el in ls_ball_center_x]))
        with open(out_fp_y, 'w') as f:
            f.write(' '.join([str(el) for el in ls_ball_center_y]))

    return ls_ball_center_x, ls_ball_center_y



class Annotator:
    def __init__(self):
        pass
    
    @staticmethod
    def annotate_video(
        processed_results, 
        out_fp,
        save_txt=True,
        limit_ball_in_table=True,
        draw_ball=True, 
        draw_person=True, 
        draw_table=True,
        fps=30,
        resolution=(1920, 1080),
        frame_limit=1e9,
    ):
        if not Path(out_fp).parent.exists():
            os.makedirs(Path(out_fp).parent)
        # define a video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_fp, fourcc, fps, resolution)
        ls_ball_center_x, ls_ball_center_y, ls_table_contour, ls_person_bb = [], [], [], []

        cnt = 0
        for frame, table_contour, ball_bb, person_bbs in processed_results:
            if draw_ball:
                if ball_bb is not None:
                    if limit_ball_in_table:
                        xmin = np.min(table_contour[:, 0])
                        xmax = np.max(table_contour[:, 0])
                    else:
                        xmin = 1e4
                        xmax = -1e4
                    ball_center_x = (ball_bb[0] + ball_bb[2])//2
                    ball_center_y = (ball_bb[1] + ball_bb[3])//2
                    if xmin < ball_center_x < xmax:
                        cv2.rectangle(frame, (ball_bb[0], ball_bb[1]), (ball_bb[2], ball_bb[3]), (0, 255, 0), 2)
                        ls_ball_center_x.append(int(ball_center_x))
                        ls_ball_center_y.append(int(ball_center_y))
                    else:
                        ls_ball_center_x.append(-1)
                        ls_ball_center_y.append(-1)
                else:
                    ls_ball_center_x.append(-1)
                    ls_ball_center_y.append(-1)

            if draw_person and person_bbs is not None:
                for person_bb in person_bbs:
                    cv2.rectangle(frame, (person_bb[0], person_bb[1]), (person_bb[2], person_bb[3]), (0, 0, 255), 2)

            if draw_table and table_contour is not None:
                cv2.polylines(frame, [table_contour], True, (255, 0, 0), 2)

            ls_table_contour.append(table_contour)
            ls_person_bb.append(person_bbs)

            out.write(frame)
            cnt += 1
            print(f'Done frame {cnt}')
            if cnt == frame_limit:
                break
        out.release()

        if draw_ball:
            print('---------------------- Drawing ball ----------------------')
            game_info = get_game_info(
                ls_ball_center_x,
                ls_ball_center_y,
                ls_table_contour,
                ls_person_bb
            )
            # reread the output video
            cap = cv2.VideoCapture(out_fp)
            # define a video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_fp.replace('.mp4', '_ball.mp4'), fourcc, fps, resolution)
            cnt = 0
            retain_cnt = 120
            retain_position = None
            while True:
                cnt += 1
                ret, frame = cap.read()
                # write frame number at top left
                cv2.putText(frame, f'Frame {cnt}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if not ret:
                    break
                if cnt in game_info:
                    cv2.circle(frame, (ls_ball_center_x[cnt], ls_ball_center_y[cnt]), 10, (0, 0, 255), -1)
                    retain_position = (ls_ball_center_x[cnt], ls_ball_center_y[cnt])
                if retain_cnt > 0:
                    retain_cnt -= 1
                    cv2.circle(frame, retain_position, 5, (0, 0, 255), -1)
                if retain_cnt == 0:
                    retain_position = None
                    retain_cnt = 120
                
                for k in game_info.keys():
                    if isinstance(k, tuple):
                        if k[0] < cnt < k[1]:
                            state = game_info[k]
                            cv2.putText(frame, state, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            break
                out.write(frame)
        
        print(f'result saved to {out_fp}')


if __name__ == '__main__':
    import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    from predictor import *

    model_fp = '/data2/tungtx2/datn/yolov8/runs/segment/train2/weights/best.pt'
    infer_cfg = {
        'source': '../samples/test_2.mp4',
        'imgsz': 640,
        'conf': 0.5,
        'stream': True
    }

    predictor = Predictor(model_fp=model_fp)
    # processed_results = predictor.predict(infer_cfg=infer_cfg)
    predictor.predict_and_save(infer_cfg, save_dir='../model_output/test_2')

    # Annotator.annotate_video(
    #     processed_results=processed_results,
    #     out_fp='../model_output/test_7_3200.mp4',
    #     frame_limit=3200,
    # )


    # get_ls_ball_center(
    #     processed_results=processed_results,
    #     frame_limit=1e9,
    #     save=True,
    #     out_fp='../model_output/test_7.txt',
    # )
