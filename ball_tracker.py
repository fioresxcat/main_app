from my_utils import *
from kalman_tracker import *
from scipy.signal import find_peaks, savgol_filter, peak_prominences, peak_widths
import json
from typing import List, Tuple, Union, Dict, Any, Optional
import math


def get_game_info(
    vid_res_dir: str,
    limit_ball_in_table: bool = True,
    table_offset: int = 0,
    return_frame_with_no_ball: bool = False
):
    """
        INPUT:
            :param vid_res_dir: Path to directory contains all txt files holding result of a frame
            :param limit_ball_in_table: if True, only return ball position when ball is in table
            :param table_offset: offset of table contour to make it smaller / bigger
            :param return_only_valid: if True, only return frames that have ball in table

        OUTPUT: a Tuple with 2 elements
                + num total frames in video
                + A dictionary contains info of all infered frames of the game video
                    game_info[frame_idx] = {
                        'ball': [cx, cy],
                        'table': tab_coord,
                        'person': person_bbs
                    }
        
        LOGIC:
    """

    game_info = {}
    ls_txt_fp = sorted(list(Path(vid_res_dir).glob('*.txt')))
    for fp_idx, txt_fp in enumerate(ls_txt_fp):
        frame_idx = int(txt_fp.stem)
        with open(txt_fp) as f:
            lines = f.readlines()
        ball_lines = filter_lines(lines, 'ball')
        table_lines = filter_lines(lines, 'table')
        person_lines = filter_lines(lines, 'person')
        event_lines = filter_lines(lines, 'event')
        serve_lines = filter_lines(lines, 'serve')

        tab_coord = [int(el) for el in table_lines[0].strip().split()[1:]] if len(table_lines) > 0 else []
        person_bbs = [[int(el) for el in line.strip().split()[1:]] for line in person_lines] if len(person_lines) > 0 else []
        if len(event_lines) > 0:
            event_line = event_lines[0]
            ev_probs = [float(el) for el in event_line.strip().split()[1:]] if len(event_lines) > 0 else []
            ev_probs = [round(el, 2) for el in ev_probs]
        else:
            ev_probs = [0, 0, 1]

        if len(serve_lines) > 0:
            serve_line = serve_lines[0]
            is_serve = int(serve_line.strip().split()[1])
        else:
            is_serve = 0

        if len(ball_lines) > 0:
            ball_line = ball_lines[0]
            cx, cy = [int(el) for el in ball_line.strip().split()[1:-1]]
            ball_score = float(ball_line.strip().split()[-1])
            xmin = -1e5
            xmax = 1e5
            if limit_ball_in_table and len(tab_coord) > 0:
                xmin = min(tab_coord[0], tab_coord[2], tab_coord[4], tab_coord[6])
                xmax = max(tab_coord[0], tab_coord[2], tab_coord[4], tab_coord[6])
                xmin -= table_offset   # widen table
                xmax += table_offset

            if xmin < cx < xmax:
                game_info[frame_idx] = {
                    'ball': [cx, cy],
                    'ball_score': ball_score,
                    'table': tab_coord,
                    'person': person_bbs,
                    'ev_probs': ev_probs,
                    'is_serve': is_serve
                }

        elif return_frame_with_no_ball:
            game_info[frame_idx] = {
                'ball': [ignore_idx, ignore_idx],
                'ball_score': 0,
                'table': tab_coord,
                'person': person_bbs,
                'ev_probs': ev_probs,
                'is_serve': is_serve
            }
    
    return len(ls_txt_fp), dict(sorted(game_info.items()))



def is_convex(poly):
    """
    Check if a polygon is convex or not.
    """
    # Compute the cross product of consecutive edges
    # pdb.set_trace()
    n = len(poly)

    signs = []
    for i in range(n):
        j = (i + 1) % n
        k = (j + 1) % n
        cross_product = np.cross(poly[j] - poly[i], poly[k] - poly[j])
        # Append the sign of the cross product to the list of signs
        signs.append(np.sign(cross_product))
    # Check if all the signs are the same
    return all(sign == signs[0] for sign in signs)



def sort_points_ccw(points: List[Tuple[int, int]]):
    # Find the center of the points
    center = tuple(sum(x) / len(x) for x in zip(*points))
    
    # Compute the angle of each point relative to the center
    angles = []
    for point in points:
        angle = math.atan2(point[1] - center[1], point[0] - center[0])
        angles.append(angle)
    
    # Sort the points based on their angle relative to the center in a counter-clockwise direction
    sorted_points = [x for _, x in sorted(zip(angles, points))]
    
    return sorted_points


def is_left(x1, y1, x2, y2, x, y):
    """
    Calculate whether the point (x, y) is to the left or right of the line
    defined by the points (x1, y1) and (x2, y2).
    """
    return (x2 - x1) * (y - y1) - (x - x1) * (y2 - y1)


def is_point_inside_polygon(point: Tuple[int, int], polygon: List[Tuple[int, int]]):
    """
    Check if a point is inside a 4-point polygon in 2D space.
    """
    x, y = point
    x0, y0 = polygon[0]
    x1, y1 = polygon[1]
    x2, y2 = polygon[2]
    x3, y3 = polygon[3]
    
    # Compute the winding number
    wn = 0
    if is_left(x1, y1, x0, y0, x, y) > 0:
        if is_left(x2, y2, x1, y1, x, y) > 0 and is_left(x3, y3, x2, y2, x, y) > 0 and is_left(x0, y0, x3, y3, x, y) > 0:
            wn = 1
    else:
        if is_left(x2, y2, x1, y1, x, y) < 0 and is_left(x3, y3, x2, y2, x, y) < 0 and is_left(x0, y0, x3, y3, x, y) < 0:
            wn = -1
    
    return wn != 0



def l2_distance(p1, p2):
    """
    Calculate the Euclidean distance between two 2D points.
    """
    x1, y1 = p1
    x2, y2 = p2
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist



class GameTracker:

    def __init__(self, fps, img_w, img_h, vid_fp, vid_res_dir, limit_ball_in_table, return_frame_with_no_ball, debug=False):
        """
            net_x: vị trí của lưới theo chiều x, ước lượng bằng cách lấy chiều rộng bàn chia đôi

        """
        self.fps = fps
        self.img_w = img_w
        self.img_h = img_h
        self.net_cx = self.img_w // 2 - 100
        self.debug = debug
        self.vid_fp = vid_fp
        self.vid_res_dir = vid_res_dir

        print('getting game info ...')
        self.total_fr, self.game_info = get_game_info(
            vid_res_dir,
            limit_ball_in_table=limit_ball_in_table,
            return_frame_with_no_ball=return_frame_with_no_ball
        )

        self.fr_indices_with_ball = sorted([fr_idx for fr_idx, info in self.game_info.items() if (info['ball'][0] >= 0 and info['ball'][1] >= 0)])
        self.tracker = KalmanBoxTracker(bbox=[0, 0, 5, 5])
        self.init_table_info()
        self.init_hyper_params()
        self.get_event_fr_indices()


    def get_event_fr_indices(self):
        # pdb.set_trace()
        self.bounce_probs = [self.game_info[fr_idx]['ev_probs'][0] for fr_idx in self.game_info.keys()]
        self.net_probs = [self.game_info[fr_idx]['ev_probs'][1] for fr_idx in self.game_info.keys()]
        self.empty_probs = [self.game_info[fr_idx]['ev_probs'][2] for fr_idx in self.game_info.keys()]

        bounce_indices, _ = find_peaks(
            self.bounce_probs, 
            distance=self.fps//4, 
            prominence=0.8, 
            width=None, 
            wlen=None
        )
        bounce_indices = bounce_indices.tolist()

        # remove indices with low prob
        self.bounce_remove_indices = []
        for idx in bounce_indices:
            nearby_probs = self.bounce_probs[max(idx - 4, 0):idx + 5]
            n_high_probs = len([prob for prob in nearby_probs if prob >= 0.6])
            if n_high_probs < 5 or all(prob < 0.9 for prob in nearby_probs):  # check nếu prob thấp
                self.bounce_remove_indices.append(idx)  # loại bỏ dự đoán này
                n_higher_probs = len([prob for prob in nearby_probs if prob >= 0.9])
                if n_high_probs >= 4 and n_higher_probs >= 2:   # check thêm 1 điều kiện nữa để cho nó hoàn lương
                    self.bounce_remove_indices.remove(idx)


        print('bounce remove indices: ', self.bounce_remove_indices)
        self.bounce_indices = [max(idx - 4 + 1, 0) for idx in bounce_indices if idx not in self.bounce_remove_indices]
        print('valid bounce indices: ', self.bounce_indices)

        net_indices, _ = find_peaks(
            self.net_probs, 
            distance=self.fps//4, 
            prominence=0.8, 
            width=None, 
            wlen=None
        )
        net_indices = net_indices.tolist()

        # remove indices with low prob
        self.net_remove_indices = []
        for idx in net_indices:
            nearby_probs = self.net_probs[max(idx - 4, 0):idx + 5]
            n_high_probs = len([prob for prob in nearby_probs if prob >= 0.6])

            if n_high_probs < 5 or all(prob < 0.9 for prob in nearby_probs):
                self.net_remove_indices.append(idx)
                n_higher_probs = len([prob for prob in nearby_probs if prob >= 0.9])
                if n_high_probs >= 4 and n_higher_probs >= 2:
                    self.net_remove_indices.remove(idx)

        # remove indices that have invalid ball pos
        for idx in net_indices:
            ls_cx, _, _ = self.get_list_coord(idx-4, idx+4, include_non_ball_frame=True)
            ls_invalid_cx = [cx for cx in ls_cx if abs(cx-self.net_cx) > 300 or (not self.tab_xmin <= cx <= self.tab_xmax)]
            if len(ls_invalid_cx) > 5:
                self.net_remove_indices.append(idx)

        print('net remove indices: ', self.net_remove_indices)
        self.net_indices = [max(idx - 4 + 1, 0) for idx in net_indices if idx not in self.net_remove_indices]
        print('valid net indices: ', self.net_indices)

        # pdb.set_trace()
        


    def init_hyper_params(self):
        """
            chỗ này chứa những tham số để phục vụ việc check rule.
        """
        # check frame with ball
        time_check_interval = 1   # 1s
        self.fr_check_interval = time_check_interval * self.fps
        self.min_fr_with_ball_percent = 0.6
        self.min_fr_without_ball_percent = 0.9

        # check travel dist to count as start
        self.min_travel_dist_thresh = self.img_w // 10

        # check velocity
        self.velocity_check_percent = 1.0   # check all velocity in the range when perform velocity check
        self.min_velocity_valid_percent = 0.5   # at least 50% of the velocity must be in valid range
        self.velocity_dict = {}     # store velocity of ball in 2 consecutive frames
        self.min_v_thresh = 5 * 120 / self.fps
        self.max_v_thresh = 50 * 120 / self.fps

        # check net hit
        self.near_net_thresh = 100 * self.img_w / 1920
        self.n_cx_near_net_thresh = 20 * self.fps / 120
        self.min_ball_one_side_percent = 0.9    # nếu có quá một số frame nào đó mà bóng nằm ở 1 bên trái hoặc phải => chạm lưới

        # check direction
        self.slope_sum_thresh = 100 * self.img_h / 1080
        
        # check valid rally
        self.min_rally_duration = 1.5       # 1,5s

        # find peaks
        self.false_neg_rate = 0.6    # cứ 100 frame có ball thì ko detect được 50 cái
        rtt_ball_time = 1       # 1s
        self.distance_x = (1 - self.false_neg_rate) * self.fps * rtt_ball_time   # là khoảng cách required giữa 2 cực trị liên tiếp. Cho vào để tránh việc noise làm cho có 2 cực trị quá sát nhau
        self.prominence_x = self.tab_w // 3
        self.width_x = self.distance_x // 3
        self.wlen_x = None
        self.rel_height_x = 1

        self.distance_y = int(10 * self.fps / 120)
        self.prominence_y = int(10 * self.img_h / 1080)
        self.width_y = self.distance_y // 3

        # smooth
        self.wlen_sm_x = 21
        self.order_sm_x = 3
        self.wlen_sm_y = 11
        self.order_sm_y = 2
        self.smooth_check_net_hit = False
        self.smooth_get_direction = True

        # tracker
        self.outlier_velocity_thresh = 100 * 120 / self.fps

        # list final rally
        self.ls_proposed_rally = []
        self.ls_rally = []


    def init_table_info(self):
        """
            OUTPUT: return table xmin, ymin, xmax, ymax, width, height, net_w
            INPUT: self.game_info đã có
            LOGIC:
              + only consider table pos when there is no ball detected
        """
        poly2cnt = {}
        for fr_idx in sorted(self.game_info.keys()):
            # only consider table pos when there is no ball detected
            if fr_idx in self.fr_indices_with_ball:
                continue

            tab_coord = tuple(self.game_info[fr_idx]['table'])
            tab_poly = np.array(tab_coord).reshape(-1, 2)
            if is_convex(tab_poly):
                if tab_coord in poly2cnt:
                    poly2cnt[tab_coord] += 1
                else:
                    poly2cnt[tab_coord] = 0
        
        tab_coord = max(poly2cnt, key=poly2cnt.get)
        tab_poly = np.array(tab_coord).reshape(-1, 2).tolist()
        pt1, pt2, pt3, pt4 = tab_poly
        tl, tr, br, bl = sort_points_ccw([pt1, pt2, pt3, pt4])

        # pdb.set_trace(header='init table info')

        # adjust
        # tl[0] -= 5
        # tl[1] -= 5
        # tr[0] += 5
        # tr[1] -= 5
        # br[0] += 5
        # br[1] += 5
        # bl[0] -= 5
        # bl[1] += 5

        self.tab_poly = [tl, tr, br, bl]
        self.real_tab_w = l2_distance(br, bl)
        self.real_tab_h = l2_distance(br, tr)
        self.net_cx = (bl[0] + br[0]) // 2
        self.tab_xmin = min(tab_coord[::2])
        self.tab_xmax = max(tab_coord[::2])
        self.tab_ymin = min(tab_coord[1::2])
        self.tab_ymax = max(tab_coord[1::2])
        self.tab_w = self.tab_xmax - self.tab_xmin
        self.tab_h = self.tab_ymax - self.tab_ymin

        offset = 30
        self.tab_xmin_extend = self.tab_xmin - offset   
        self.tab_xmax_extend = self.tab_xmax + offset
        self.tab_ymin_extend = self.tab_ymin - offset
        self.tab_ymax_extend = self.tab_ymax + offset
        self.tab_poly_extend = [
            [self.tab_xmin - offset, self.tab_ymin - offset],
            [self.tab_xmax + offset, self.tab_ymin - offset],
            [self.tab_xmax + offset, self.tab_ymax + offset],
            [self.tab_xmin - offset, self.tab_ymax + offset]
        ]


    def generate_hightlight(self):
        """
            Generate hightlight based on all self.history
        """
        print('------------ Generating proposed rally ------------')
        is_started = False
        started_fr = None
        end_fr = None
        for index, fr_idx in enumerate(self.fr_indices_with_ball):
            if is_started:
                is_end, check_info = self.check_end(index, fr_idx)
                if is_end:
                    is_started = False
                    # end_fr = fr_idx + self.fr_check_interval if not check_info['reason'] == 'reach_end' else self.total_fr
                    end_fr = fr_idx if not check_info['reason'] == 'reach_end' else self.total_fr

                    end_fr = min(end_fr, self.total_fr)
                    print(f'Ball ends at frame {end_fr},', check_info['reason'])
                    self.ls_proposed_rally.append([started_fr, end_fr])
                    started_fr, end_fr = None, None
                    if check_info['reason'] == 'reach_end':
                        break
            else:
                is_started, check_info = self.check_start(index, fr_idx, debug=True)
                if is_started:
                    started_fr = fr_idx
                    print(f'\nBall starts at frame {started_fr},', check_info['reason'])

        # print('\n------------ Filter valid rally ------------')
        # self.ls_rally = []
        # for rally in self.ls_proposed_rally:
        #     # if rally[0] != 9658:
        #     #     continue
        #     print('\nChecking rally: ', rally)
        #     if self.check_valid_rally(rally[0], rally[1], is_debug=True):
        #         self.ls_rally.append(rally)
        # self.ls_rally = sorted(self.ls_rally, key=lambda x: x[0])

        self.ls_rally = self.ls_proposed_rally
        return self.ls_rally
    


    def check_start(self, index, fr_idx, debug=False):
        """
            check 2 rules:
              + trong 120 frame tiếp theo, ít nhất 60% trong số đó là có bóng
              + Check ball trajectory:
                Ball trajectory được lấy từ fr_idx -> fr_idx + 120 (120 là fr_check_interval) (chỉ lấy những frame có bóng)
                + check velocity: có ít nhất 70% vận tốc tức thời (pixel/frame) nằm trong ngưỡng cho phép (min_v < v < max_v)
                + check min travel distance: max_cx và min_cx của quả bóng phải lệch nhau ít nhất 1 ngưỡng (đang để là 200 với ảnh đầu vào 1920x1080)
        """
        info = {}

        # ------------------------ check min frame with ball ----------------------
        pass_min_fr_with_ball = self.check_min_fr_with_ball(index, fr_idx)
        
        # ------------------ check velocity ------------------
        ls_cx, ls_cy, fr_indices = self.get_list_coord(fr_idx, fr_idx+self.fr_check_interval, include_non_ball_frame=False)
        is_valid_velocity = self.check_min_n_valid_velocity(ls_cx, ls_cy, fr_indices)
        
        # ------------------ check max travel distance ------------------
        smooth = True
        if smooth:
            ls_cx = self.smooth_x(ls_cx)
        is_valid_min_travel_dist = abs(max(ls_cx)-min(ls_cx)) > self.min_travel_dist_thresh

        # ------------------ check serve predict result ---------------------
        is_serve = self.check_serve(index, fr_idx)

        # if (pass_min_fr_with_ball and is_valid_velocity and is_valid_min_travel_dist) or is_serve:
        #     info['result'] = 'pass'
        #     info['reason'] = 'pass min_fr_with_ball, velocity, min travel distance test'
        #     return True, info
        # else:
        #     if not pass_min_fr_with_ball:
        #         info['result'] = 'fail'
        #         info['reason'] = 'fail min_fr_with_ball test'
        #         return False, info

        #     if not is_valid_velocity:
        #         info['result'] = 'fail'
        #         info['reason'] = 'fail velocity test'
        #         return False, info

        #     if not is_valid_min_travel_dist:
        #         info['result'] = 'fail'
        #         info['reason'] = 'fail min travel distance test'
        #         return False, info


        if is_serve:
            info['result'] = 'pass'
            info['reason'] = 'pass serve detect'
            return True, info
        else:
            info['result'] = 'fail'
            info['reason'] = 'fail serve detect'
            return False, info
        

    def check_min_fr_with_ball(self, index, fr_idx):
        """
            :param index: index of the frame in the list of has-ball frames
            :param fr_idx: real index of the frame
        """
        # minimum required number of frame with ball to be count as ball start
        min_fr_with_ball = int(self.min_fr_with_ball_percent * self.fr_check_interval)

        # nếu ở gần cuối rồi => cho thành False vì ko thể nào start ở gần cuối được
        if index + min_fr_with_ball >= len(self.fr_indices_with_ball):
            return False
        
        # điều kiện chính
        if self.fr_indices_with_ball[index+min_fr_with_ball] < fr_idx + self.fr_check_interval:
            return True
        
        return False
    

    def check_min_fr_without_ball(self, index, fr_idx):
        # maximum number of frame with ball. If exceed this number => will not be count as end
        max_fr_with_ball = int((1-self.min_fr_without_ball_percent) * self.fr_check_interval)

        # nếu là frame cuối có bóng => cho end
        if index == len(self.fr_indices_with_ball) - 1:
            return True
        
        # nếu là frame gần cuối có bóng => cho thành False để nó chạy đến frame cuối
        if index + max_fr_with_ball >= len(self.fr_indices_with_ball):
            return False
        
        # điều kiện chính: nếu phải ra ngoài khoảng fr_check_interval mới đạt đủ max_fr_with_ball => đúng là trong khoảng check có rất nhiều frame ko có bóng => True
        if self.fr_indices_with_ball[index + max_fr_with_ball] > fr_idx + self.fr_check_interval:
            return True
        
        return False


    def check_min_n_valid_velocity(self, ls_cx, ls_cy, ls_fr_idx):
        """
            ls_cx: list of x coord
            ls_cy: list of y coord
            ls_fr_idx: frame indices corresponding to each coord
            MUST BE IN ASCENDING ORDER
        """
        assert len(ls_cx) == len(ls_cy) == len(ls_fr_idx), "The 3 lists must have same length"

        # giới hạn việc check vận tốc ở 1 phần frame đầu thôi (thường vận tốc bóng sẽ lớn ở những frame đầu, sau đó giảm dấn)
        ball_trajectory = list(zip(ls_cx, ls_cy, ls_fr_idx))
        ls_velocity = []
        for i in range(len(ball_trajectory)-1):
            # get info of 2 consecutive frames
            start_fr_info = ball_trajectory[i]
            end_fr_info = ball_trajectory[i+1]

            # calc num frame diff
            frame_diff = abs(end_fr_info[2] - start_fr_info[2])

            # calc velocity from start frame to end frame and stores in a dict
            if (start_fr_info[2], end_fr_info[2]) not in self.velocity_dict:
                x1, y1, x2, y2 = start_fr_info[0], start_fr_info[1], end_fr_info[0], end_fr_info[1]
                coord_diff = math.sqrt((x1-x2)**2 + (y1-y2)**2)
                v = round(coord_diff / frame_diff, 2)
                self.velocity_dict[(start_fr_info[2], end_fr_info[2])] = v  # đơn vị: pixel / frame
            ls_velocity.append(self.velocity_dict[(start_fr_info[2], end_fr_info[2])])
        
        n_valid = len([v for v in ls_velocity if self.max_v_thresh > v > self.min_v_thresh])
        print(f'valid velocity: {n_valid}/{len(ball_trajectory)}') if self.debug else None
        return n_valid > self.min_velocity_valid_percent * len(ls_cx)



    def check_end(self, index, fr_idx, debug=False):
        info = {}

        if fr_idx + self.fr_check_interval >= self.total_fr:
            info['reason'] = 'reach_end'
            info['result'] = 'passed'

        if self.check_min_fr_without_ball(index, fr_idx):
            info['reason'] = 'pass check min fr without ball'
            info['result'] = 'passed'
            return True, info
        
        return False, info
    
    
    def check_valid_rally(self, start_fr_idx, end_fr_idx, is_debug=False):
        """
            LOGIC: Check 2 conditions
            - rally must be longer than min_rally_duration
            - rally must have x-axis extrema
              + để lọc ra những lần người chơi trả bóng cho đối phương phát
              + tuy nhiên lại ko bắt được những pha giao bóng hỏng (ko có extrema_x)
        """
        # -------------------------- check duration ----------------------------
        if abs(end_fr_idx - start_fr_idx) < self.fps * self.min_rally_duration:
            print(f'duration check FAIL: {end_fr_idx - start_fr_idx} < {self.fps * self.fps * self.min_rally_duration}') if is_debug else None
            return False
        else:
            print(f'duration check OK: {end_fr_idx - start_fr_idx}') if is_debug else None
        
        # -------------------------- check x-axis extrema --------------------------
        # pdb.set_trace()
        ls_cx, ls_cy, fr_indices = self.get_list_coord(start_fr_idx, end_fr_idx)
        _, maxima_x, minima_x, _ = self.get_extrema_x(ls_cx,  fr_indices, smooth=True, return_type='seperate')
        if len(maxima_x) == 0 and len(minima_x) == 0:
            print(f'number of extrema check FAIL: maxima_x: {maxima_x}, minima_x: {minima_x}') if is_debug else None
            return False
        else:
            print(f'number of extrema check OK: maxima_x: {maxima_x}, minima_x: {minima_x}') if is_debug else None

        return True
    

    def get_list_coord(self, start_idx, end_idx, include_non_ball_frame=False):
        ls_cx, ls_cy, fr_indices = [], [], []
        for fr_idx in range(start_idx, end_idx+1):
            if fr_idx not in self.fr_indices_with_ball and not include_non_ball_frame:
                continue
            if fr_idx in self.game_info:
                ls_cx.append(self.game_info[fr_idx]['ball'][0])
                ls_cy.append(self.game_info[fr_idx]['ball'][1])
                fr_indices.append(fr_idx)
        return ls_cx, ls_cy, fr_indices

 
    # def get_extrema_x(self, start_fr_idx, end_fr_idx, smooth=True, return_fr_idx=True, return_type='all'):
    def get_extrema_x(
        self, 
        ls_cx, 
        fr_indices, 
        smooth=True, 
        return_fr_idx=True, 
        return_type='all',
        distance=-1,
        prominence=-1,
        width=-1,
        wlen=-1,
        rel_height=-1,
    ):
        if smooth:
            ls_cx = self.smooth_x(ls_cx, wlen=None, order=None)

        # find peaks
        maxima, max_prop = find_peaks(
            ls_cx, 
            distance=distance if distance != -1 else self.distance_x, 
            prominence=prominence if prominence != -1 else self.prominence_x, 
            width=width if width != -1 else self.width_x, 
            wlen=wlen if wlen != -1 else None, 
            rel_height=rel_height if rel_height != -1 else self.rel_height_x
        )
        maxima = maxima.tolist()
        minima, min_prop = find_peaks(
            -np.array(ls_cx), 
            distance=distance if distance != -1 else self.distance_x, 
            prominence=prominence if prominence != -1 else self.prominence_x, 
            width=width if width != -1 else self.width_x, 
            wlen=wlen if wlen != -1 else None, 
            rel_height=rel_height if rel_height != -1 else self.rel_height_x
        )
        minima = minima.tolist()

        # convert from peak indices to frame indices
        if return_fr_idx:
            maxima = [fr_indices[el] for el in maxima]
            minima = [fr_indices[el] for el in minima]

        prop = {
            'maxima': max_prop,
            'minima': min_prop
        }

        if return_type == 'all':
            return ls_cx, sorted(minima+maxima), prop
        elif return_type == 'seperate':
            return ls_cx, sorted(minima), sorted(maxima), prop
        else:
            raise ValueError('Return type not supported')


    # def get_maxima_y(self, start_fr_idx, end_fr_idx, return_fr_idx=True, smooth=True):
    def get_maxima_y(self, ls_cy, fr_indices, return_fr_idx=True, smooth=True):
        if smooth:
            ls_cy = self.smooth_y(ls_cy, wlen=None, order=None)

        # find peaks
        maxima, prop = find_peaks(
            ls_cy, 
            distance=self.distance_y, 
            prominence=self.prominence_y, 
            width=self.width_y, 
            wlen=None
        )
        maxima = maxima.tolist()

        # convert from peak indices to frame indices
        if return_fr_idx:
            maxima = [fr_indices[el] for el in maxima]

        return ls_cy, maxima, prop


    def get_rally_info(self, start_idx, end_idx, save_dir=None):
        """
            Return info about the rally:
                + list ball bounces
                + why the rally end ?
                + num turns in the rally
        """
        info = {}

        # get coord 
        ls_cx, ls_cy, fr_indices = self.get_list_coord(start_idx, end_idx)
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_fn = f'{start_idx}_{end_idx}_original.jpg'
            self.plot_ball_pos(ls_cx, ls_cy, fr_indices, save_path=os.path.join(save_dir, save_fn))

            ls_sm_cx = self.smooth_x(ls_cx, wlen=None, order=None)
            ls_sm_cy = self.smooth_y(ls_cy, wlen=None, order=None)
            save_fn = f'{start_idx}_{end_idx}_smooth.jpg'
            self.plot_ball_pos(ls_sm_cx, ls_sm_cy, fr_indices, save_path=os.path.join(save_dir, save_fn))
        
        # # correct outlier coord
        # ls_cx, ls_cy, fr_indices = self.correct_list_coord(ls_cx, ls_cy, fr_indices)
        # if save_dir is not None:
        #     save_fn = f'{start_idx}_{end_idx}_corrected.jpg'
        #     self.plot_ball_pos(ls_cx, ls_cy, fr_indices, save_path=os.path.join(save_dir, save_fn))

        # get manual bounces
        _, maxima_y, _ = self.get_maxima_y(ls_cy, fr_indices, smooth=False)
        manual_bounce_indices = self.get_valid_bounces(maxima_y)

        # get predicted bounces
        bounce_indices = [fr_idx for fr_idx in self.bounce_indices if fr_idx >= start_idx and fr_idx <= end_idx]
        bounce_indices = self.get_valid_bounces(bounce_indices)

        # # add bounce in predicted but not in manual
        # diff_limit = 10  # 10 frame
        # for idx in bounce_indices:
        #     is_duplicate = False
        #     for manual_idx in manual_bounce_indices:
        #         if abs(idx - manual_idx) <= diff_limit:
        #             is_duplicate = True
        #             break
        #     if not is_duplicate:
        #         manual_bounce_indices.append(idx)

        # info['bounce_indices'] = manual_bounce_indices   # using both manual_bounce_indices and predicted_bounce_indices
        info['bounce_indices'] = bounce_indices   # not using manual_bounce_indices


        # get number of turns
        _, list_extrema_x, _ = self.get_extrema_x(ls_cx, fr_indices, smooth=True)
        net_indices = [fr_idx for fr_idx in self.net_indices if fr_idx >= start_idx and fr_idx <= end_idx]
        info['n_turns'] = len(list_extrema_x) + 1
        # info['n_turns'] = len(net_indices)   # try to use num_net_hit instead of len(list_extrema_x) + 1
        info['net_indices'] = net_indices

        
        # -------------------------- check end_reason and winner -----------------------
        
        res = self.AI_check_end_reason_and_winner(start_idx, end_idx, bounce_indices, net_indices)
        info['end_reason'] = res['end_reason']
        info['winner'] = res['winner']

        return info
    

    def manual_check_end_reason_and_winner(self, start_idx, end_idx):
        info = {}
        ls_cx, ls_cy, fr_indices = self.get_list_coord(start_idx, end_idx)
        _, list_extrema_x, _ = self.get_extrema_x(ls_cx, fr_indices, smooth=True)
        range2check = [list_extrema_x[-1], end_idx]
        print('range2check', range2check)
        
        direction = self.get_direction(range2check[0], range2check[1])
        is_net_hit, winner = self.check_net_hit(range2check[0], range2check[1], direction)
        if is_net_hit:
            info['end_reason'] = 'net_hit'
            info['winner'] = winner
        else:
            # check with manual bounce
            ls_cx, ls_cy, fr_indices = self.get_list_coord(range2check[0], range2check[1])
            _, bounce_indices, _ = self.get_maxima_y(ls_cy, fr_indices, smooth=True)
            if len(bounce_indices) > 0:
                bounce_idx = bounce_indices[0]
                bounce_cx, bounce_cy = self.game_info[bounce_idx]['ball']
                if abs(bounce_cx-self.net_cx) < self.near_net_thresh:   # nếu bounce ở gần lưới => khả năng cao là bóng chạm lưới bật lên
                    info['end_reason'] = 'ball_out, bounce near net'
                    info['winner'] = 'left' if direction == 'r2l' else 'right'
                else:   # bóng bounce xa lưới 
                    if self.tab_xmin < bounce_cx < self.tab_xmax or bounce_cy > self.tab_ymax:   # bóng bounce trong bàn
                        info['end_reason'] = 'good_ball'
                        info['winner'] = 'right' if direction == 'r2l' else 'left'
                    else:   # bounce ngoài bàn
                        info['end_reason'] = 'ball_out, bounce out of table'
                        info['winner'] = 'left' if direction == 'r2l' else 'right'

        return info
        

    def AI_check_end_reason_and_winner(self, start_idx, end_idx, final_bounce_indices, final_net_indices) -> Dict[str, Any]:
        if len(final_bounce_indices) == 0:
            return {'end_reason': 'unknown', 'winner': 'unknown'}
        
        last_bounce_idx = final_bounce_indices[-1]
        last_bounce_pos = self.game_info[last_bounce_idx]['ball']
        bounce_side = 'left' if last_bounce_pos[0] < self.net_cx else 'right'
        ls_cx, ls_cy, fr_indices = self.get_list_coord(last_bounce_idx, end_idx, include_non_ball_frame=False)

        winner = 'right' if bounce_side == 'left' else 'left'
        _, extrema_x, _ = self.get_extrema_x(
            ls_cx, 
            fr_indices,
            smooth=False,
            distance=None,
            prominence=self.distance_x//5,
            width=None,
            wlen=None,
            rel_height=None,
        )
        if len(extrema_x) == 0:   # nếu bóng trôi tiếp ko nảy lại => good ball from the other side
            return {
                'end_reason': 'good ball',
                'winner': winner
            }
        else:
            num_over_net = len([fr_idx for fr_idx in final_net_indices if fr_idx >= last_bounce_idx and fr_idx <= end_idx])
            if num_over_net > 0:    # nếu có bóng qua lưới
                return {
                    'end_reason': 'ball out',
                    'winner': winner
                }
            else:   # nếu ko có bóng qua lưới
                return {
                    'end_reason': 'net hit',
                    'winner': winner
                }
            

    def hybrid_check_end_reason_and_winner(self, start_idx, end_idx, final_bounce_indices, final_net_indices):
        if len(final_bounce_indices) == 0:
            return {'end_reason': 'unknown', 'winner': 'unknown'}
        
        last_bounce_idx = final_bounce_indices[-1]
        last_bounce_pos = self.game_info[last_bounce_idx]['ball']
        bounce_side = 'left' if last_bounce_pos[0] < self.net_cx else 'right'
        ls_cx, ls_cy, fr_indices = self.get_list_coord(last_bounce_idx, end_idx, include_non_ball_frame=False)

        winner = 'right' if bounce_side == 'left' else 'left'
        
        num_over_net = len([fr_idx for fr_idx in final_net_indices if fr_idx >= last_bounce_idx and fr_idx <= end_idx])
        if num_over_net > 0:    # nếu có bóng qua lưới
            info = {
                'end_reason': 'ball_out',
                'winner': winner
            }
        
        else:  # nếu ko có bóng qua lưới
            n_ball_in_table = len([cx for cx in ls_cx if self.tab_xmin_extend <= cx <= self.tab_xmax_extend])
            if n_ball_in_table / len(ls_cx) > 0.5:  # nếu có nhiều bóng trong bàn
                # xet them logic de phong truong hop ko detect duoc su kien net
                n_ball_left = len([cx for cx in ls_cx if cx < self.net_cx])
                n_ball_right = len([cx for cx in ls_cx if cx > self.net_cx])
                min_side_percent = min(n_ball_left, n_ball_right) / len(ls_cx)

                if min_side_percent > 0.3:  # neu ball o ca 2 phia cua luoi => ball out
                    info = {
                        'end_reason': 'ball_out',
                        'winner': winner
                    }
                else:  # neu ball chi o 1 phia cua luoi => net hit
                    info = {
                        'end_reason': 'net_hit',
                        'winner': winner
                    }

            else:   # neu co it ball in table, do la một pha bóng tốt (ko đỡ lại được)
                info = {
                    'end_reason': 'good_ball',
                    'winner': winner
                }





    # def plot_ball_pos(self, start_idx, end_idx, smooth=False, save_path='test.jpg'):
    def plot_ball_pos(self, ls_cx, ls_cy, fr_indices, use_frame_idx_as_x=True, save_path='test.jpg'):
        ls_cx, extrema_x, _ = self.get_extrema_x(ls_cx, fr_indices, return_fr_idx=use_frame_idx_as_x, smooth=True)
        ls_cy, maxima_y, _ = self.get_maxima_y(ls_cy, fr_indices, return_fr_idx=use_frame_idx_as_x, smooth=True)

        # Create a figure and axis object
        fig, ax = plt.subplots(figsize=(20, 7))
        
        # plot ball position
        if use_frame_idx_as_x:
            ax.plot(fr_indices, ls_cx, color='blue', label='x')
            ax.plot(fr_indices, ls_cy, color='orange', label='y')
        else:
            ax.plot(list(range(len(ls_cx))), ls_cx, color='blue', label='x')
            ax.plot(list(range(len(ls_cy))), ls_cy, color='orange', label='y')

        # set xtick, ytick
        if use_frame_idx_as_x:
            ax.set_xticks(list(range(fr_indices[0], fr_indices[-1]+1, 50)))
        else:
            ax.set_xticks(list(range(0, len(ls_cx), 50)))

        # plot extrema
        for index in extrema_x:
            ax.axvline(x=index, color='r', linestyle='-')
        for index in maxima_y:
            ax.axvline(x=index, color='g', linestyle='--')

        ax.set_title("Line Plot of X and Y coord")
        ax.set_xlabel("Frame Index")
        ax.set_ylabel("Value")
        ax.legend()

        fig.savefig(save_path)


    def check_net_hit(self, start_idx, end_idx, direction):
        """
            Check trong 1 turn qua lại (chứ không phải là cả rally, chỉ là 1 turn trong rally đó)
            start_idx là frame bắt đầu turn, end_idx là frame kết thúc turn

            Các trường hợp cần bắt:
            - bóng từ 1 bên -> nảy vào lưới -> đi ngang
            - bóng từ 1 bên -> nảy vào lưới -> nảy lại
        """
        is_net_hit, winner = False, None
        ls_cx, ls_cy, fr_indices = self.get_list_coord(start_idx, end_idx, include_non_ball_frame=False)
        if self.smooth_check_net_hit:
            ls_cx = self.smooth_x(ls_cx)

        # pdb.set_trace()
        # nếu bóng đập vào luwois rồi nảy lại => last_extrema_x sẽ là lúc bóng đập vào lưới
        if abs(ls_cx[0]-self.net_cx) <= self.near_net_thresh:
            winner = 'left' if direction == 'l2r' else 'right'
            is_net_hit = True
            return is_net_hit, winner
        
        # check: nếu có nhiều frame mà bóng loanh quanh cạnh lưới => đúng là bóng chạm lưới
        n_cx_near_net = len([cx for cx in ls_cx if abs(cx-self.net_cx) < self.near_net_thresh])
        if n_cx_near_net > self.n_cx_near_net_thresh:
            winner = 'left' if direction == 'r2l' else 'right'
            is_net_hit = True
            return is_net_hit, winner
        
        # check: nếu chủ yếu các frame nằm ở 1 bên của lưới => đúng là bóng chạm lưới
        n_ball_right = len([cx for cx in ls_cx if cx >= self.net_cx])
        n_ball_left = len([cx for cx in ls_cx if cx <= self.net_cx])
        if max(n_ball_right, n_ball_left) > self.min_ball_one_side_percent * len(fr_indices):
            winner = 'left' if direction == 'r2l' else 'right'
            is_net_hit = True
            return is_net_hit, winner

        # find 2 consecutive extrema with largest distance 
        _, extrema_x, _ = self.get_extrema_x(ls_cx, fr_indices, smooth=False)       
        extrema_x = fr_indices[:1] + extrema_x + fr_indices[-1:]  # append first and last frame
        max_extrema_dist = -1
        max_prev, max_next = None, None
        for fr_idx in extrema_x[:-1]:
            index = fr_indices.index(fr_idx)
            prev_extrema = ls_cx[index]
            next_extrema = ls_cx[index]
            if abs(prev_extrema-next_extrema) > max_extrema_dist:
                max_prev = prev_extrema
                max_next = next_extrema
                max_extrema_dist = abs(prev_extrema-next_extrema)
        
        # check: nếu 1 trong 2 extrema gần lưới => đúng là bóng chạm lưới
        if min(abs(max_prev-self.net_cx), abs(max_next-self.net_cx)) < self.near_net_thresh:
            winner = 'left' if direction == 'r2l' else 'right'
            is_net_hit = True
            return is_net_hit, winner

        
        return is_net_hit, winner
    

    def check_serve(self, index, fr_idx):
        """
            check if in the next 10 frames (15 frames interval), there are at least 7 frames that is serve
        """
        if self.game_info[fr_idx]['is_serve'] == 0:
            return False
        
        check_limit = 10 # check 10 next frames is serve
        fr_interval = 15
        min_num_is_serve = 7

        frame_indices_to_check = [fr_idx]
        for i in range(check_limit):
            fr_idx += fr_interval
            frame_indices_to_check.append(fr_idx)
        
        num_is_serve = len([fr_idx for fr_idx in frame_indices_to_check if fr_idx in self.game_info and self.game_info[fr_idx]['is_serve'] == 1])

        # TODO
        # if serve, check if the next following frames  actually contain enough ball, if not, it's not serve

        return num_is_serve >= min_num_is_serve





    def get_direction(self, start_idx, end_idx):
        ls_cx, ls_cy, fr_indices = self.get_list_coord(start_idx, end_idx, include_non_ball_frame=False)
        if self.smooth_get_direction:
            ls_cx = self.smooth_x(ls_cx, wlen=None, order=None)
        slopes = [ls_cx[i+1] - ls_cx[i] for i in range(len(ls_cx)-1)]
        slopes_sum = sum(slopes)
        if slopes_sum > self.slope_sum_thresh:
            return "l2r"
        elif slopes_sum < -self.slope_sum_thresh:
            return "r2l"
        else:
            return "still"
        
        
    
    def smooth_x(self, ls_cx, wlen=None, order=None):
        if wlen is None:
            wlen = self.wlen_sm_x
        if order is None:
            order = self.order_sm_x
        return savgol_filter(ls_cx, window_length=wlen, polyorder=order) if len(ls_cx) > wlen else ls_cx
    

    def smooth_y(self, ls_cy, wlen=None, order=None):
        if wlen is None:
            wlen = self.wlen_sm_y
        if order is None:
            order = self.order_sm_y
        return savgol_filter(ls_cy, window_length=wlen, polyorder=order) if len(ls_cy) > wlen else ls_cy


    def get_range2check(self, start_idx, end_idx):
        """
            INPUT:
                start_idx is start of rally
                end_idx is end of rally
            OUTPUT:
                what range should be checked for end_reason

            LOGIC: có 4 loại quỹ đạo có thể
            + bóng chạm người 1 -> đánh sang người 2 ko đỡ được -> chỉ check range cuối
            + bóng chạm người 1 -> đánh sang người 2 đỡ được nhưng lại nảy ra ngoài -> chỉ check range cuối
            + bóng chạm người 1 -> đánh sang người 2 đỡ rúc lưới -> chỉ check range cuối
            + bóng chạm người 1 -> đánh sang người 2 đỡ rúc lưới và nảy lại người 2 -> phải check từ 2 range cuối
        """
        pass


    def correct_list_coord(self, ls_cx, ls_cy, fr_indices, save_dir=None, strategy=1, trk_thresh=10, use_last_valid_frame=True):
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            ls_frame_fp = []
            for img_fp in Path(self.vid_res_dir).glob('*.jpg'):
                img_idx = int(img_fp.stem)
                if img_idx in fr_indices:
                    ls_frame_fp.append(img_fp)

        trk_update_cnt = 0
        ls_invalid = []
        last_valid_index = 0
        for i in range(1, len(ls_cx)):
            # predict using kalman filter
            pred_bb = self.tracker.predict()[0]

            # get detected coord
            cx, cy, fr_idx = ls_cx[i], ls_cy[i], fr_indices[i]

            # if not 1002 <= fr_idx <= 1005:
            #     continue
            
            if use_last_valid_frame:
                prev_cx, prev_cy, prev_fr_idx = ls_cx[last_valid_index], ls_cy[last_valid_index], fr_indices[last_valid_index]
            else:
                prev_cx, prev_cy, prev_fr_idx = ls_cx[i-1], ls_cy[i-1], fr_indices[i-1]
            dist = l2_distance((prev_cx, prev_cy), (cx, cy))
            fr_diff = fr_idx - prev_fr_idx

            # pdb.set_trace()

            if dist / fr_diff > self.outlier_velocity_thresh:   # nếu vị trí detect được là outlier
                ls_invalid.append(fr_idx)
                print(f'coord at frame {fr_idx} is OUTLIER with velocity {dist/fr_diff}')
                if strategy == 1:  # use last_valid_frame coord
                    ls_cx[i] = ls_cx[last_valid_index]
                    ls_cy[i] = ls_cy[last_valid_index]
                elif strategy == 2:  # use predict coord
                    if trk_update_cnt < trk_thresh:  # tracker chưa đủ tin tưởng
                        ls_cx[i] = ls_cx[last_valid_index]
                        ls_cy[i] = ls_cy[last_valid_index]
                    else:
                        ls_cx[i] = pred_bb[0]   # gán cho nó bằng vị trí predict bằng tracker
                        ls_cy[i] = pred_bb[1]

            else:    # nếu vị trí detect được là valid
                print(f'coord at frame {fr_idx} is VALID with velocity {dist/fr_diff}')
                bb = (cx-2, cy-2, cx+2, cy+2)
                self.tracker.update(bb)  # update tracker
                trk_update_cnt += 1
                last_valid_index = i

            if save_dir is not None:
                img_fp = [fp for fp in ls_frame_fp if int(fp.stem) == fr_idx][0]
                img = cv2.imread(str(img_fp))
                # draw a circle arounf center of pred_bb
                pred_cx, pred_cy = int(pred_bb[0]+pred_bb[2])//2, int(pred_bb[1]+pred_bb[3])//2
                cv2.circle(img, (pred_cx, pred_cy), 10, (0, 255, 0), -1)
                cv2.imwrite(str(Path(save_dir) / img_fp.name), img)

        print('ls_invalid: ', ls_invalid)
        return ls_cx, ls_cy, fr_indices


    def get_valid_bounces(self, bounce_indices):
        ls_ball_pos = [self.game_info[idx]['ball'] for idx in bounce_indices]
        # bounce_indices = [bounce_indices[i] for i, pos in enumerate(ls_ball_pos) if is_point_inside_polygon(pos, self.tab_poly)]
        bounce_indices = [bounce_indices[i] for i, pos in enumerate(ls_ball_pos) if is_point_inside_polygon(pos, self.tab_poly_extend)]

        return bounce_indices


    def save_game_insights(self):
        """
            Mỗi frame sẽ có các trường thông tin sau trong info
            + state: is_in_rally or not_in_rally
            + is_bounce: True or False
            + bounce_pos_to_draw: 
            + end_info: 
              + n_turns
              + reason:
              + winner:
            + left_score: a number
            + right_score: a number
        """
        game_insights = {}
        left_score, right_score = 0, 0

        # -------------- for frames in rally ---------------------
        for rally in self.ls_rally:
            start_idx, end_idx = rally
            rally_info = self.get_rally_info(start_idx, end_idx)
            for fr in range(start_idx, end_idx+1):
                fr_info = {}

                # write state
                fr_info['state'] = 'is_in_rally'

                # write ball info
                fr_info['ball_pos'] = self.game_info[fr]['ball'] if fr in self.fr_indices_with_ball else None
                fr_info['ball_score'] = self.game_info[fr]['ball_score']


                # write bounce info
                if fr in rally_info['bounce_indices']:
                    fr_info['is_bounce'] = True
                    fr_info['bounce_pos_to_draw'] = fr_info['ball_pos']
                else:
                    fr_info['is_bounce'] = False
                    fr_info['bounce_pos_to_draw'] = None
                
                # write score
                fr_info['left_score'] = left_score
                fr_info['right_score'] = right_score

                # write end info if have
                if fr == end_idx:
                    fr_info['end_info'] = {}
                    fr_info['end_info']['n_turns'] = rally_info['n_turns']
                    fr_info['end_info']['reason'] = rally_info['end_reason']
                    fr_info['end_info']['winner'] = rally_info['winner']
                    if rally_info['winner'] == 'left':
                        left_score += 1
                    else:
                        right_score += 1
                    fr_info['end_info_to_draw'] = fr_info['end_info']
                else:
                    fr_info['end_info'] = None
                    fr_info['end_info_to_draw'] = None
                
                # save info to game_insights
                game_insights[fr] = fr_info
        
        ls_fr_in_rally = sorted(list(game_insights.keys()))

        # -------------- for frames not in rally ---------------------
        for fr in range(1, self.total_fr+1):
            if fr in game_insights:
                continue

            # find nearest frame with score
            ls_prev_fr_in_rally = sorted([fr_idx for fr_idx in ls_fr_in_rally if fr_idx < fr])
            if len(ls_prev_fr_in_rally) > 0:
                nearest_fr_in_rally = ls_prev_fr_in_rally[-1]
                current_left_score = game_insights[nearest_fr_in_rally]['left_score']
                current_right_score = game_insights[nearest_fr_in_rally]['right_score']
            else:
                current_left_score = 0
                current_right_score = 0

            fr_info = {}
            fr_info['state'] = 'not_in_rally'
            fr_info['ball_pos'] = self.game_info[fr]['ball'] if fr in self.fr_indices_with_ball else None
            fr_info['ball_score'] = self.game_info[fr]['ball_score']
            fr_info['is_bounce'] = False
            fr_info['bounce_pos_to_draw'] = None
            fr_info['end_info'] = None
            fr_info['end_info_to_draw'] = None
            fr_info['left_score'] = current_left_score
            fr_info['right_score'] = current_right_score

            game_insights[fr] = fr_info


        # adjust bounce_pos_to_draw
        for fr in ls_fr_in_rally:
            fr_info = game_insights[fr]
            # if fr_info['is_bounce']:
            #     for fr2adjust in range(fr+1, min(fr+120, self.total_fr+1)):
            #         if game_insights[fr2adjust]['bounce_pos_to_draw'] is None:
            #             game_insights[fr2adjust]['bounce_pos_to_draw'] = fr_info['bounce_pos_to_draw']
            
            if fr_info['end_info'] is not None:
                for fr2adjust in range(fr+1, min(fr+120, self.total_fr+1)):
                    if game_insights[fr2adjust]['end_info_to_draw'] is None:
                        game_insights[fr2adjust]['end_info_to_draw'] = fr_info['end_info_to_draw']

        self.game_insights = game_insights
        return game_insights
    

    def annotate_vid(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        cap = cv2.VideoCapture(self.vid_fp)
        total_fr_number = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_w, out_h = 1080, 640
        cnt = 0
        first_mark, offset = 30, 30
        # pdb.set_trace()
        while True:
            success, frame = cap.read()
            if not success:
                break
            cnt += 1

            fr_info = self.game_insights[cnt]

            # write frame number at topleft
            cv2.putText(frame, f'Frame {cnt}', (10, first_mark), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # write score
            left_score, right_score = fr_info['left_score'], fr_info['right_score']
            cv2.putText(frame, f'SCORE: {left_score} - {right_score}', (10, first_mark+1*offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # write state
            state = fr_info['state']
            cv2.putText(frame, f'STATE: {state}', (10, first_mark+2*offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # plot ball pos if have
            ball_pos = fr_info['ball_pos']
            if ball_pos is not None:
                # draw a red circle at ball_pos
                cv2.circle(frame, ball_pos, 10, (0, 0, 255), -1)
                ball_score = fr_info['ball_score']
                cv2.putText(frame, f'{ball_score:.2f}', (ball_pos[0]-10, ball_pos[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # plot bounce pos if have
            bounce_pos = fr_info['bounce_pos_to_draw']
            if bounce_pos is not None:
                # draw a green circle at ball_pos
                cv2.circle(frame, bounce_pos, 10, (0, 255, 0), -1)

            # write end_info if have
            end_info = fr_info['end_info_to_draw']
            if end_info is not None:
                cv2.putText(frame, f'END INFO:', (10, first_mark+3*offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, f'Winner: {end_info["winner"]}', (10, first_mark+4*offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, f'Reason: {end_info["reason"]}', (10, first_mark+5*offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, f'Number of turns: {end_info["n_turns"]}', (10, first_mark+6*offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


            # write table coord
            tl, tr, br, bl = self.tab_poly
            # draw polylines on frame
            cv2.polylines(frame, [np.array([tl, tr, br, bl])], True, (0, 255, 0), 2)


            frame = cv2.resize(frame, (out_w, out_h))
            cv2.imwrite(os.path.join(save_dir, f'{cnt:05d}.jpg'), frame)
            print(f'Done frame {cnt}/{total_fr_number}')



if __name__ == '__main__':
    pass