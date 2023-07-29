import torch
import time
import pdb
import os
import cv2
import pickle
import os
from yolo_predictor import YoloPredictor
from centernet_onnx import CenternetOnnx
from event_cls_onnx import EventClsOnnx
from serve_detect_onnx import ServeDetectOnnx
from event_cls_3d_onnx import EventCls3DOnnx
from collections import deque, namedtuple
import numpy as np
from PIL import Image



def save_ball_prediction(img, ball_pos, out_fp):
    """
        img: shape 3 x 512 x 512, float32
        ball_pos: tuple(cx, cy), normalized, opencv coord (not numpy coord)
        out_fp: fp to save
    """
    img = np.transpose(img, (1, 2, 0))
    img = (img*255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cx, cy = ball_pos
    abs_cx, abs_cy = int(cx*img.shape[1]), int(cy*img.shape[0])
    img = cv2.circle(img, center=(abs_cx, abs_cy), radius=8, color=(255, 0, 0), thickness=1)
    cv2.imwrite(out_fp, img)


def save_yolo_prediction(img, person_bbs, table_poly, ball_pos, out_fp):
    """
        img: img with original shape (1920, 1080)
        person_bbs: np.arrray, shape num_player x 4
        table_poly: np.array, shape 4 x 2
    """
    person_bbs = person_bbs * np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    person_bbs = person_bbs.astype(np.int32)

    table_poly = table_poly * np.array([img.shape[1], img.shape[0]])
    table_poly = table_poly.astype(np.int32)

    ball_pos = (ball_pos[0] * img.shape[1], ball_pos[1] * img.shape[0])
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for bb in person_bbs:
        x1, y1, x2, y2 = bb
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
    img = cv2.drawContours(img, [table_poly], -1, (0, 255, 0), 2)
    img = cv2.circle(img, center=(int(ball_pos[0]), int(ball_pos[1])), radius=8, color=(0, 0, 255), thickness=1)
    cv2.imwrite(out_fp, img)



class Predictor:
    def __init__(
        self,
        yolo_detector,
        ball_detector,
        event_detector,
        serve_classifier,
        debug_dir
    ):
        self.yolo_detector = yolo_detector
        self.ball_detector = ball_detector
        self.event_detector = event_detector
        self.serve_classifier = serve_classifier
        self.debug_dir = debug_dir

        if self.event_detector is not None:
            self.event_detector.crop_size = self.event_detector.crop_size

        self.orig_h, self.orig_w = (1080, 1920)
        self._init_deque()
    

    def _init_deque(self):
        self.running_ball_frames = deque(maxlen=self.ball_detector.n_input_frames)
        self.running_ev_frames = deque(maxlen=self.event_detector.n_input_frames) if self.event_detector is not None else None
        self.running_ev_ball_pos = deque(maxlen=self.event_detector.n_input_frames) if self.event_detector is not None else None
        self.running_serve_frames = deque(maxlen=self.serve_classifier.n_input_frames) if self.serve_classifier is not None else None

    
    def predict_video(
        self,
        video_fp,
        out_dir,
    ):
        """
            all coords are returned as abs coords, in original frame shape: (1080, 1920)
        """
        os.makedirs(out_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_fp)
        # get total number of frame
        total_frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_cnt = 0
        serve_frame_interval = 0
        suc = True
        while suc:
            start = time.perf_counter()
            suc, frame = cap.read()
            if not suc:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame_cnt += 1

            # predict ball, rgb image
            ball_pos, ball_score = None, None
            self.update_running_ball_frames(frame)
            if len(self.running_ball_frames) == self.running_ball_frames.maxlen and self.ball_detector is not None:
                norm_ball_pos, ball_score = self.predict_ball()   # ball_pos is never None, only (-1, -1) or valid pos
                ball_pos = np.array(norm_ball_pos) * np.array([self.orig_w, self.orig_h])
                ball_pos = tuple(ball_pos.astype(np.int32))

            # predict player, table, bgr image
            person_bbs, table_poly, yolo_ball_pos = None, None, None
            if self.yolo_detector is not None:
                person_bbs, table_poly, yolo_ball_pos = self.predict_person_and_table(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))   # yolo needs BGR image
                if person_bbs is not None:
                    person_bbs = np.array(person_bbs) * np.array([self.orig_w, self.orig_h, self.orig_w, self.orig_h])
                    person_bbs = person_bbs.astype(np.int32)
                if table_poly is not None:
                    table_poly = np.array(table_poly) * np.array([self.orig_w, self.orig_h])
                    table_poly = table_poly.astype(np.int32)

            # predict event, rgb image
            event=None
            if ball_pos is not None and self.event_detector is not None:
                self.running_ev_ball_pos.append(ball_pos)
                self.running_ev_frames.append(frame)
                if len(self.running_ev_ball_pos) == self.running_ev_ball_pos.maxlen:
                    n_valid_ball_pos = len([pos for pos in self.running_ev_ball_pos if pos[0]>0 and pos[1]>0])
                    if n_valid_ball_pos >= self.running_ev_ball_pos.maxlen * 2 // 3:
                        event = self.predict_event()

            # predict serve, rgb image
            is_serve = None
            if self.serve_classifier is not None:
                serve_frame_interval += 1
                if serve_frame_interval == 15:
                    self.running_serve_frames.append(frame)
                    serve_frame_interval = 0
                    if len(self.running_serve_frames) == self.running_serve_frames.maxlen:
                        print('Detecting serve ...')
                        is_serve = self.predict_serve()

            # write result
            self.write_result(
                out_dir,
                self.frame_cnt,
                ball_pos=ball_pos,
                ball_score=ball_score,
                person_bbs=person_bbs,
                table_poly=table_poly,
                event=event,
                is_serve=is_serve
            )
            duration = time.perf_counter() - start
            print(f'Done frame {self.frame_cnt}/{total_frame_cnt} in {duration:.2f}s')


    def update_running_ball_frames(self, frame):
        """
            input: orig frame shape (1920, 1080, 3)
            output: np array shape (3, 512, 512)
        """
        frame = cv2.resize(frame, (512, 512))
        self.running_ball_frames.append(frame)    


    def predict_ball(self):
        """
            get data from self.running_ball_frames and infer
            img in self.ball_frames are rgb, shape (512, 512, 3)
        """
        if self.ball_detector is None:
            return (-1, -1)
        
        # preprocess
        imgs = np.concatenate(self.running_ball_frames, axis=-1)
        imgs = np.transpose(imgs, (2, 0, 1))
        imgs = (imgs / 255.).astype(np.float32)
        imgs = np.expand_dims(imgs, axis=0)

        # infer
        batch_pos, scores = self.ball_detector.predict(imgs)

        # postprocess
        pos = batch_pos[0]
        score = scores[0][0]
        # pdb.set_trace()
        if pos[0] == 0 and pos[1] == 0:
            cx, cy = -1, -1
        else:
            cx = float(pos[0])
            cy = float(pos[1])
        
        # visualize
        if cx >= 0 and cy >= 0:
            save_dir = os.path.join(self.debug_dir, 'ball')
            os.makedirs(save_dir, exist_ok=True)
            img = self.running_ball_frames[-1]
            abs_cx, abs_cy = int(cx * img.shape[1]), int(cy * img.shape[0])
            img = cv2.circle(img, (abs_cx, abs_cy), 5, (255, 0, 0), 2)
            img = cv2.putText(img, f'{score:.2f}', (max(0, abs_cx-10), max(0, abs_cy-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(save_dir, f'{self.frame_cnt}.jpg'), img)

        return (cx, cy), score


    def predict_person_and_table(self, frame):
        """
            frame is original frame shape (1080, 1920, 3)
        """
        person_bbs, table_bb, ball_pos = self.yolo_detector.predict_img(frame)

        # normalize
        if ball_pos is not None:
            cx, cy = ball_pos
            cx = cx / frame.shape[1]
            cy = cy / frame.shape[0]
            ball_pos = (cx, cy)

        # normalize
        if person_bbs is not None:
            person_bbs = person_bbs / np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])

        # normalize
        if table_bb is not None:
            table_bb = table_bb / np.array([frame.shape[1], frame.shape[0]])

        return person_bbs, table_bb, ball_pos
    

    def predict_event(self):
        """
            get data from self.running_ev_frames and self.running_ev_ball_pos and infer
            self.running_ev_frames: orig frame shape (1080, 1920, 3)
            self.running_ev_ball_pos: list of (cx, cy)
        """
        # preprocess frame
        median_cx = int(np.median([pos[0] for pos in self.running_ev_ball_pos if pos[0] > 0 and pos[1] > 0]))
        median_cy = int(np.median([pos[1] for pos in self.running_ev_ball_pos if pos[0] > 0 and pos[1] > 0]))
        xmin = max(0, median_cx - self.event_detector.crop_size[0]//2)
        xmax = min(median_cx + self.event_detector.crop_size[0]//2, 1920)
        ymin = max(0, median_cy - self.event_detector.crop_size[1]//3)
        ymax = min(median_cy + self.event_detector.crop_size[1]*2//3, 1080)

        cropped_frames = []
        save_dir = os.path.join(self.debug_dir, 'event', f'{self.frame_cnt-8}-{self.frame_cnt}')
        os.makedirs(save_dir, exist_ok=True)
        for i, frame in enumerate(self.running_ev_frames):
            try:
                cropped_frame = frame[ymin:ymax, xmin:xmax, :]
                abs_pos = self.running_ev_ball_pos[i]
                if self.event_detector.mask_red_ball and all(el > 0 for el in abs_pos):
                    cropped_pos = (abs_pos[0] - xmin, abs_pos[1] - ymin)
                    cropped_frame = cv2.circle(cropped_frame, cropped_pos, self.event_detector.ball_radius, (0, 0, 255), -1)
                cropped_frame = cv2.resize(cropped_frame, self.event_detector.input_size)   # shape (128, 320, 3)
            except Exception as e:
                print(e)
                pdb.set_trace()
            cropped_frames.append(cropped_frame)
            cv2.imwrite(os.path.join(save_dir, f'{self.frame_cnt-8+i}.jpg'), cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2BGR))

        # # preprocess pos
        # pos = np.array(self.running_ev_ball_pos).astype(np.float32)
        # pos = np.expand_dims(pos, axis=0)   # shape (1, 9, 2)

        # infer
        event = self.event_detector.predict(cropped_frames)
        return event
    

    def predict_serve(self):
        """
            self.running_serve_frames is list of original frames read from video
        """
        is_serve = self.serve_classifier.predict(self.running_serve_frames, return_probs=False)
        is_serve = is_serve[0][0]
        return is_serve

    

    def write_result(
        self,
        out_dir: str,
        frame_cnt: int,
        ball_pos: tuple =None,
        ball_score: float =None,
        person_bbs: np.array =None,    # shape num_players x 4
        table_poly: np.array =None,      # shape 4 x 2
        event: int =None,
        is_serve: int = None
    ):
        """
            all coords are normalized
        """
        out_fp = os.path.join(out_dir, f'{frame_cnt:06d}.txt')
        with open(out_fp, 'w') as f:
            if table_poly is not None:
                x1, y1, x2, y2, x3, y3, x4, y4 = table_poly.reshape(-1).tolist()
                f.write(f'0 {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}\n')
            if person_bbs is not None:
                for bb in person_bbs:
                    xmin, ymin, xmax, ymax = bb
                    f.write(f'1 {xmin} {ymin} {xmax} {ymax}\n')
            if ball_pos is not None:
                cx, cy = ball_pos
                f.write(f'2 {cx} {cy} {ball_score:02f}\n')
            if event is not None:
                bounce_prob, net_prob, empty_prob = event.reshape(-1).tolist()
                f.write(f'3 {bounce_prob} {net_prob} {empty_prob}\n')
            if is_serve is not None:
                f.write(f'4 {int(is_serve)}\n')


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    yolo = YoloPredictor(
        model_fp='models/best_yolov8n_segment_train6.pt',
        imgsz=640,
        conf=0.5
    )
    
    centernet = CenternetOnnx(
        # onnx_path='models/ball_detect_centernet_3_frames_epoch51.onnx',
        # onnx_path='models/ball_detect_centernet_exp71_epoch40_add_no_ball_frame.onnx',
        # onnx_path='models/ball_detect_add_pos_pred_weight_exp78_epoch43.onnx',
        # onnx_path='models/exp80_centernet_add_pos_pred_weight_add_no_ball_frame.onnx',
        # onnx_path='models/exp81_ep38_multiball_add_no_ball_frame_add_pos_pred_weight.onnx',
        onnx_path='models/exp85_ep43_centernet_no_asl_3_frames.onnx',
        n_input_frames=3,
        decode_by_area=False,
        conf_thresh=0.5
    )

    # event_cls = EventClsOnnx(
    #     onnx_path='models/exp4_ce_loss_event_cls_epoch36.onnx',
    #     n_input_frames=9
    # )

    event_cls_3d = EventCls3DOnnx(
        # onnx_path='models/event_cls_3d_exp2_ep19.onnx',
        onnx_path='models/event_detector_epoch87_mask_blue_ball.onnx',
        n_input_frames=9
    )

    serve_classifier = ServeDetectOnnx(
        onnx_path='models/serve_detect_ep26.onnx',
        n_input_frames=15
    )

    ls_vid_fn = [
        'test_4.mp4', 
        'test_7.mp4', 
        'test_3.mp4', 

        'test_1.mp4',
        'test_5.mp4', 
        'test_6.mp4', 
        'test_2.mp4'
    ]
    for vid_fn in ls_vid_fn:
        print(f'processing {vid_fn}...')
        predictor = Predictor(
            yolo_detector=yolo,
            # yolo_detector=None,

            ball_detector=centernet,

            event_detector=event_cls_3d,
            # event_detector=None,

            serve_classifier=serve_classifier,
            # serve_classifier=None,

            # debug_dir=f'debug/{vid_fn[:-4]}_added_no_ball_frame_decode_area/',
            # debug_dir=f'debug/{vid_fn[:-4]}_not_added/,
            debug_dir=f'debug_final/{vid_fn[:-4]}',
            # debug_dir=f'debug/{vid_fn[:-4]}_multiball_added_no_ball_frame_add_pos_pred_weight/'

        )

        video_fp = f'samples/{vid_fn}'
        predictor.predict_video(
            video_fp=video_fp,
            # out_dir=f'results/{vid_fn[:-4]}_added_no_ball_frame_decode_area',
            # out_dir=f'results/{vid_fn[:-4]}_not_added',
            out_dir=f'results_final/{vid_fn[:-4]}',
            # out_dir=f'results/{vid_fn[:-4]}_multiball_added_no_ball_frame_add_pos_pred_weight_real',
        )