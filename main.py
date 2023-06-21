import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# from pathlib import Path
# import cv2
# import numpy as np
# import shutil
# import torch
# from PIL import Image, ImageDraw
# import sys
# from ultralytics import YOLO
# from typing import List, Tuple, Union
# import json
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.signal import find_peaks, savgol_filter
# from typing import List, Tuple
from my_utils import *
from ball_tracker import *


if __name__ == '__main__':
    tracker = GameTracker(
        fps = 120, 
        img_w=1920,
        img_h=1080,
        vid_fp='../samples/test_4.mp4',
        vid_res_dir='results/test_4',
        limit_ball_in_table=False,
        return_frame_with_no_ball=True,
        debug=False,
    )

    tracker.generate_hightlight()
    print()
    print('proposed rally: '.capitalize(), tracker.ls_proposed_rally)
    print('final rally: '.capitalize(), tracker.ls_rally)

    print('-------------------------- Inspecting valid rally --------------------------')
    for rally in tracker.ls_rally:
        if rally[0] != 176:
            continue
        ls_cx, ls_cy, fr_indices = tracker.get_list_coord(rally[0], rally[1])
        print('rally: ', rally)
        print('list extrema x: ', tracker.get_extrema_x(ls_cx, fr_indices)[1])
        info = tracker.get_rally_info(rally[0], rally[1], save_dir='debug/test_7_rally_info')
        print('info: ', info)
        print('\n')

    # gi = tracker.save_game_insights()
    # # sort gi by keys
    # gi = {k: v for k, v in sorted(gi.items(), key=lambda item: item[0])}
    # pdb.set_trace()

    # tracker.annotate_vid(save_dir='results/test_4_annotate_main')