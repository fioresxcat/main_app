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
from yolo_predictor import *
import sys

label2id = {
    'table': 0,
    'person': 1,
    'ball': 2,
    'event': 3,
    'serve': 4,
}
id2label = {v:k for k, v in label2id.items()}
ignore_idx = int(-1e4)



def compute_area(row: np.array):
    xmin, ymin, xmax, ymax = row.tolist()
    w = xmax - xmin
    h = ymax - ymin
    return w * h



class Annotator:
    def __init__(self):
        pass
    
    @staticmethod
    def annotate_video(
        res_dir, 
        out_fp,
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
        for frame, table_contour, ball_coord, person_bbs in processed_results:
            if draw_ball:
                if ball_coord is not None:
                    if limit_ball_in_table:
                        xmin = np.min(table_contour[:, 0])
                        xmax = np.max(table_contour[:, 0])
                    else:
                        xmin = 1e4
                        xmax = -1e4
                    ball_center_x = ball_coord[0]
                    ball_center_y = ball_coord[1]
                    if xmin < ball_center_x < xmax:
                        cv2.circle(frame, (int(ball_center_x), int(ball_center_y)), 10, (0, 0, 255), -1)
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
                    cv2.circle(frame, retain_position, 10, (0, 0, 255), -1)
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


def get_size(var):
    size_in_bytes = sys.getsizeof(var)

    if size_in_bytes < 1024:
        size = f"{size_in_bytes} bytes"
    elif size_in_bytes < 1024 * 1024:
        size = f"{size_in_bytes / 1024:.2f} KB"
    elif size_in_bytes < 1024 * 1024 * 1024:
        size = f"{size_in_bytes / (1024 * 1024):.2f} MB"
    else:
        size = f"{size_in_bytes / (1024 * 1024 * 1024):.2f} GB"

    return size


def filter_lines(lines, class_name):
    return [line for line in lines if line.strip().split()[0] == str(label2id[class_name])]  # find line starts with table class id


def is_monotonous(ls):
    threshold = 0.9
    slopes = [ls[i+1] - ls[i] for i in range(len(ls)-1)]
    num_pos = len([s for s in slopes if s > 0])
    num_neg = len([s for s in slopes if s < 0])
    return num_pos / len(slopes) > threshold or num_neg / len(slopes) > threshold



# def get_game_info(
#     vid_res_dir: str,
#     limit_ball_in_table: bool = True,
#     table_offset: int = 0,
#     return_frame_with_no_ball: bool = False
# ):
#     """
#         INPUT:
#             :param vid_res_dir: Path to directory contains all txt files holding result of a frame
#             :param limit_ball_in_table: if True, only return ball position when ball is in table
#             :param table_offset: offset of table contour to make it smaller / bigger
#             :param return_only_valid: if True, only return frames that have ball in table

#         OUTPUT: a Tuple with 2 elements
#                 + num total frames in video
#                 + A dictionary contains info of all infered frames of the game video
#                     game_info[frame_idx] = {
#                         'ball': [cx, cy],
#                         'table': tab_coord,
#                         'person': person_bbs
#                     }
        
#         LOGIC:
#     """

#     game_info = {}
#     ls_txt_fp = sorted(list(Path(vid_res_dir).glob('*.txt')))
#     for fp_idx, txt_fp in enumerate(ls_txt_fp):
#         frame_idx = int(txt_fp.stem)
#         # if frame_idx == 1092:
#         #     pdb.set_trace()
#         with open(txt_fp) as f:
#             lines = f.readlines()
#         ball_lines = filter_lines(lines, 'ball')
#         table_lines = filter_lines(lines, 'table')
#         person_lines = filter_lines(lines, 'person')

#         tab_coord = [int(el) for el in table_lines[0].strip().split()[1:]] if len(table_lines) > 0 else []
#         person_bbs = [[int(el) for el in line.strip().split()[1:]] for line in person_lines] if len(person_lines) > 0 else []
        
#         if len(ball_lines) > 0:
#             ball_info = ball_lines[0]
#             cx, cy = [int(el) for el in ball_info.strip().split()[1:]]
#             xmin = -1e5
#             xmax = 1e5
#             if limit_ball_in_table and len(tab_coord) > 0:
#                 xmin = min(tab_coord[0], tab_coord[2], tab_coord[4], tab_coord[6])
#                 xmax = max(tab_coord[0], tab_coord[2], tab_coord[4], tab_coord[6])
#                 xmin -= table_offset   # widen table
#                 xmax += table_offset

#             if xmin < cx < xmax:
#                 game_info[frame_idx] = {
#                     'ball': [cx, cy],
#                     'table': tab_coord,
#                     'person': person_bbs
#                 }

#         elif return_frame_with_no_ball:
#             game_info[frame_idx] = {
#                 'ball': [ignore_idx, ignore_idx],
#                 'table': tab_coord,
#                 'person': person_bbs
#             }
    
#     return len(ls_txt_fp), dict(sorted(game_info.items()))



def convert_extrema_to_frame_indices(extrema_indices, fr_indices):
    return [fr_indices[el] for el in extrema_indices]

if __name__ == '__main__':
    import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    from yolo_predictor import *

    model_fp = '/data2/tungtx2/datn/yolov8/runs/segment/train3/weights/best.pt'
    infer_cfg = {
        'source': '../samples/test_7.mp4',
        'imgsz': 640,
        'conf': 0.5,
        'stream': True
    }

    predictor = Predictor(model_fp=model_fp)
    # processed_results = predictor.predict(infer_cfg=infer_cfg)
    predictor.predict_and_save(infer_cfg, save_dir='../model_output/test_7_regen')

    # Annotator.annotate_video(
    #     processed_results=processed_results,
    #     out_fp='../model_output/test_2_regen/test_2_annotated.mp4',
    #     frame_limit=int(1e9),
    # )


    # get_ls_ball_center(
    #     processed_results=processed_results,
    #     frame_limit=1e9,
    #     save=True,
    #     out_fp='../model_output/test_7.txt',
    # )
