import os
from pathlib import Path
import cv2
from PIL import Image
from ultralytics import YOLO
from ultralytics.yolo.engine.results import Results
import numpy as np
import torch
from my_utils import *
import math
import pdb
from typing import List, Tuple, Union



def compute_area(row: np.array):
    xmin, ymin, xmax, ymax = row.tolist()
    w = xmax - xmin
    h = ymax - ymin
    return w * h


def compute_dist(pt1: tuple, pt2: tuple) -> float:
    """
    Computes the Euclidean distance between two points in 2D space.

    INPUT:
        :param pt1: A tuple representing the (x, y) coordinates of the first point.
        :type pt1: tuple[float, float]
        :param pt2: A tuple representing the (x, y) coordinates of the second point.
        :type pt2: tuple[float, float]
        :return: The distance between the two points.
        :rtype: float
    """
    return math.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)



def approx_tab_contour(tab_contour):
    alpha = 0.01
    epsilon = alpha * cv2.arcLength(tab_contour, True)
    tab_contour = cv2.approxPolyDP(tab_contour, epsilon, True)
    tab_contour = tab_contour.reshape(-1, 2)
    while len(tab_contour) > 4 and alpha < 0.1:
        alpha += 0.01
        epsilon = alpha * cv2.arcLength(tab_contour, True)
        tab_contour = cv2.approxPolyDP(tab_contour, epsilon, True)
        tab_contour = tab_contour.reshape(-1, 2)
        # assert len(table_contour) == 4, f"approx should have 4 points, but got {len(table_contour)}"
    return tab_contour


def find_nearest_ball(ball_bbs, last_ball_pos):
    ls_pos = [((bb[0] + bb[2]) // 2, (bb[1] + bb[3]) // 2) for bb in ball_bbs]
    ls_dist = [compute_dist(pos, last_ball_pos) for pos in ls_pos]
    min_idx = np.argmin(ls_dist)
    return ball_bbs[min_idx]


class YoloPredictor:
    def __init__(
        self, 
        model_fp,
        imgsz,
        conf
    ):
        self.model = YOLO(model_fp)
        self.imgsz = imgsz
        self.conf = conf
        self.id2label = self.model.names
        self.label2id = {v: k for k, v in self.id2label.items()}


    def predict(self, infer_cfg, use_last_ball_pos=False):
        raw_res = self.model.predict(**infer_cfg)
        last_ball_pos = None
        for res in raw_res:
            processed_res = self.process_res(res, last_ball_pos)
            _, _, ball_pos, _ = processed_res
            if use_last_ball_pos:
                last_ball_pos = ball_pos
            yield processed_res
    

    def predict_img(self, img):
        res = self.model.predict(source=img, imgsz=self.imgsz, conf=self.conf)
        frame, table_contour, ball_pos, person_bbs = self.process_res(
            res[0],
            last_ball_pos=None,
            remove_umpire=True
        )
        return person_bbs, table_contour, ball_pos
    

    def predict_and_save(self, infer_cfg, save_dir):
        os.makedirs(os.path.join(save_dir, 'results'), exist_ok=True)
        processed_res_stream = self.predict(infer_cfg)
        frame_idx = 0
        for frame, table_contour, ball_pos, person_bbs in processed_res_stream:
            frame_idx += 1
            save_fp = os.path.join(save_dir, 'results', f'{frame_idx:05d}.txt')
            with open(save_fp, 'w') as f:
                if table_contour is not None and len(table_contour) == 4:
                    cv2.drawContours(frame, [table_contour], -1, (0, 255, 0), 2)
                    x1, y1, x2, y2, x3, y3, x4, y4 = table_contour.reshape(-1).tolist()
                    f.write(f'0 {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}\n')
                if person_bbs is not None:                
                    for bb in person_bbs:
                        xmin, ymin, xmax, ymax = bb
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                        f.write(f'1 {xmin} {ymin} {xmax} {ymax}\n')
                if ball_pos is not None:
                    cx, cy = ball_pos
                    cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)
                    f.write(f'2 {cx} {cy}\n')
            
            cv2.putText(frame, f'{frame_idx:05d}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)       # draw frame_idx at top left
            cv2.imwrite(os.path.join(save_dir, 'results', f'{frame_idx:05d}.jpg'), frame)
            print(f'saved {frame_idx:05d}')


    def process_res(self, res: Results, last_ball_pos=None, remove_umpire=True):
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
        if res.masks is None or res.boxes is None:
            return res.orig_img, None, None, None
        
        boxes = res.boxes.data.cpu().numpy()
        segments = res.masks.xy

        table_contour = self.find_table(boxes, segments)
        person_bbs = self.find_person(boxes, remove_umpire=remove_umpire)
        ball_pos = self.find_ball(boxes, last_ball_pos=last_ball_pos)
        
        
        return res.orig_img, table_contour, ball_pos, person_bbs
    

    def find_table(self, boxes, segments):
        """
            OUTPUT: np.array(shape=(4, 2)): 4 góc của table
            LOGIC:
            + if there is 1 table, return that table
            + if there are more than 1 tables, return one with largest area
        """
        tab_indices = np.where(boxes[:, -1] == self.label2id['table'])[0]
        if len(tab_indices) <= 0:
            return None
        
        if len(tab_indices) == 1:
            tab_idx = tab_indices[0]
        else:
            tab_contours = [segments[idx].astype(int) for idx in tab_indices]
            tab_areas = [cv2.contourArea(contour) for contour in tab_contours]
            tab_idx = np.argmax(tab_areas)
        tab_contour = segments[tab_idx].astype(int)
        tab_contour = approx_tab_contour(tab_contour)
        if len(tab_contour) != 4:
            return None
        return tab_contour
    

    def find_person(self, boxes: np.array, remove_umpire=True) -> np.array:
        """
            OUTPUT: np.array(shape(n, 4)): bounding box của n người
            LOGIC:
            + find bbs that are person in boxes
            + if there are 2 or fewer than 2 person, return the person_bbs as-is
            + if there are more than 2 person:
              + if remove_umpire: return only 2 boxes with largest area
              + if not remove_umpire: return the person_bbs as-is
        """
        ls_person_idx = np.where(boxes[:, -1] == self.label2id['person'])[0]
        if len(ls_person_idx) <= 0:
            return None
        
        person_bbs = boxes[ls_person_idx, :4].astype(int)
        if len(person_bbs) > 2 and remove_umpire:   # let's remove the umpire
            person_areas = np.apply_along_axis(compute_area, axis=1, arr=person_bbs)
            max_area_indices = np.argsort(person_areas)[-2:]    # take 2 indices what has the largest area
            person_bbs = person_bbs[max_area_indices]
        return person_bbs.astype(int)
    

    def find_ball(self, boxes: np.array, last_ball_pos:Tuple[int, int]=None) -> Tuple[int, int]:
        """
            OUTPUT: tọa độ tâm bóng dưới dạng 1 tuple
            LOGIC:
            + nếu có 1 ball -> return ball đó
            + nếu có 2 ball trở lên
              + nếu dùng last_ball_pos -> return vị trí ball mà gần với last_ball_coord nhất
                Với giả sử rằng vị trí ball ở 2 frame liên tiếp có ball là gần nhâu
              + nếu ko dùng last_ball_pos -> return ball có confidence_score cao nhất
        """
        ball_indices = np.where(boxes[:, -1] == self.label2id['ball'])[0]
        if len(ball_indices) <= 0:
            return None
        
        if len(ball_indices) == 1:
            ball_idx = ball_indices[0]
            ball_bb = boxes[ball_idx]
        else: 
            ball_bbs = boxes[ball_indices]
            if last_ball_pos is None:
                ball_idx = np.argmax(ball_bbs[:, 4])      # find ball with max confidence
            else:  
                ball_bb = find_nearest_ball(ball_bbs, last_ball_pos)
        cx, cy = int(ball_bb[0] + ball_bb[2]) // 2, int(ball_bb[1] + ball_bb[3]) // 2
        ball_pos = (cx, cy)
        return ball_pos