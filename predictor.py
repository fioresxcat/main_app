import os
from pathlib import Path
import cv2
from PIL import Image
from ultralytics import YOLO
from ultralytics.yolo.engine.results import Results
import numpy as np
from my_utils import *


class Predictor(object):
    def __init__(self, model_fp):
        self.model = YOLO(model_fp)

    def predict(self, infer_cfg):
        raw_res = self.model.predict(**infer_cfg)
        for res in raw_res:
            yield self.process_res(res)
    
    def predict_and_save(self, infer_cfg, save_dir):
        os.makedirs(os.path.join(save_dir, 'results'), exist_ok=True)
        raw_res = self.model.predict(**infer_cfg)
        frame_idx = 0
        for res in raw_res:
            frame_idx += 1
            processed_res = self.process_res(res)
            _, table_contour, ball_bb, person_bbs = processed_res
            xmin = np.min(table_contour[:, 0])
            xmax = np.max(table_contour[:, 0])
            ymin = np.min(table_contour[:, 1])
            ymax = np.max(table_contour[:, 1])
            save_fp = os.path.join(save_dir, 'results', f'{frame_idx:05d}.txt')
            with open(save_fp, 'w') as f:
                if table_contour is not None:
                    f.write(f'0 {xmin} {ymin} {xmax} {ymax}\n')
                if person_bbs is not None:                
                    for bb in person_bbs:
                        xmin, ymin, xmax, ymax = bb
                        f.write(f'1 {xmin} {ymin} {xmax} {ymax}\n')
                if ball_bb is not None:
                    f.write(f'2 {ball_bb[0]} {ball_bb[1]} {ball_bb[2]} {ball_bb[3]}\n')
            print(f'saved {frame_idx:05d}.txt')

    def process_res(self, res: Results, remove_umpire=True):
        """
            r is a single Result object. Each contains the following attributes:
                boxes: ultralytics.yolo.engine.results.Boxes object
                    data: tensor([[429.6754, 541.3815, 449.2957, 560.9753,   0.6765,   0.0000]])
                    cls: tensor([0.])
                    conf: tensor([0.6765])
                    data: tensor([[429.6754, 541.3815, 449.2957, 560.9753,   0.6765,   0.0000]])
                    id: None
                    is_track: False
                    orig_shape: tensor([1080, 1920])
                    shape: torch.Size([1, 6])
                    xywh: tensor([[439.4855, 551.1785,  19.6203,  19.5938]])
                    xywhn: tensor([[0.2289, 0.5104, 0.0102, 0.0181]])
                    xyxy: tensor([[429.6754, 541.3815, 449.2957, 560.9753]])
                    xyxyn: tensor([[0.2238, 0.5013, 0.2340, 0.5194]])
                    
                keypoints: None
                keys: ['boxes']
                masks: None
                    data <class 'torch.Tensor'>:
                        shape: (number_of_segmented_objects, height, width)
                        each element along axis 0 is a [0, 1] mask for the corresponding object
                        height, width is the same as the input image (resized to 640)
                    orig_shape <class 'tuple'>: original shape of the input image before resizing
                    segments <class 'list'>:
                        len: number_of_segmented_objects
                        shape of each element: np.array, (number_of_segmented_points_for_each_object, 2)
                    shape <class 'torch.Size'>
                    xy <class 'list'>:
                        the same as segments
                    xyn <class 'list'>:
                        the same as segments, but normalized to [0, 1]

                names: {0: 'table', 1: 'person'}
                orig_img: np.array, the origin img
                orig_shape: (1080, 1920)
                path: '/data2/tungtx2/datn/yolov8/test.jpg'
                probs: None
                speed: {'preprocess': 1.0058879852294922, 'inference': 812.0763301849365, 'postprocess': 34.50298309326172}
        """
        id2label = res.names
        label2id = {v: k for k, v in id2label.items()}
        boxes = res.boxes.data.cpu().numpy()
        segments = res.masks.xy
        ball_bb, table_contour, person_bbs = None, None, None

        # ----------------------- find table -----------------------
        ls_table_idx = np.where(boxes[:, -1] == label2id['table'])[0]
        if len(ls_table_idx) > 0:
            if len(ls_table_idx) == 1:
                table_idx = ls_table_idx[0]
            else:
                # find idx with maximum area
                max_idx = None
                max_area = 0
                for idx in ls_table_idx:
                    table_contour = segments[idx].astype(int)
                    area = cv2.contourArea(table_contour)
                    if area > max_area:
                        max_area = area
                        max_idx = idx
                table_idx = max_idx
            table_contour = segments[table_idx].astype(int)
            # table_bb = cv2.boundingRect(table_contour)
            alpha = 0.01
            epsilon = alpha * cv2.arcLength(table_contour, True)
            table_contour = cv2.approxPolyDP(table_contour, epsilon, True)
            table_contour = table_contour.reshape(-1, 2)
            while len(table_contour > 4) and alpha < 0.1:
                alpha += 0.01
                epsilon = alpha * cv2.arcLength(table_contour, True)
                table_contour = cv2.approxPolyDP(table_contour, epsilon, True)
                table_contour = table_contour.reshape(-1, 2)
            # assert len(table_contour) == 4, f"approx should have 4 points, but got {len(table_contour)}"

        # ----------------------- find person -----------------------
        ls_person_idx = np.where(boxes[:, -1] == label2id['person'])[0]
        if len(ls_person_idx) > 0:
            person_bbs = boxes[ls_person_idx, :4].astype(int)
            if len(person_bbs) > 2 and remove_umpire:   # let's remove the umpire
                person_areas = np.apply_along_axis(compute_area, axis=1, arr=person_bbs)
                min_area_idx = np.argmin(person_areas)
                person_bbs = np.delete(person_bbs, min_area_idx, axis=0)
            person_bbs = person_bbs.astype(int)

        # ----------------------- find ball -----------------------
        ls_ball_idx = np.where(boxes[:, -1] == label2id['ball'])[0]
        if len(ls_ball_idx) > 0:
            if len(ls_ball_idx) == 1:
                ball_idx = ls_ball_idx[0]
            else:  # if there is more than one ball
                # find ball with max confidence
                ls_ball_bb = boxes[ls_ball_idx]
                ball_idx = np.argmax(ls_ball_bb[:, 4])
            ball_bb = boxes[ball_idx, :4].astype(int)

        return res.orig_img, table_contour, ball_bb, person_bbs