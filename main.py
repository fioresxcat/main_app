import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from my_utils import *
import pdb
from ball_tracker import *


def get_info(game_info, fr_idx, info_type):
    for idx in range(fr_idx-10, fr_idx+10):
        print(idx, game_info[idx][info_type])


def get_date():
    import datetime

    today = datetime.datetime.now()
    date_string = today.strftime("%d-%m-%Y")
    return date_string


def get_debug_dir(root_dir, description=None):
    os.makedirs(root_dir, exist_ok=True)
    exp_nums = [int(subdir[3:]) if '_' not in subdir else int(subdir.split('_')[0][3:]) for subdir in os.listdir(root_dir)]
    max_exp_num = max(exp_nums) if len(exp_nums) > 0 else 0
    exp_name = f'exp{max_exp_num+1}' if description is None else f'exp{max_exp_num+1}_{description}'
    return os.path.join(root_dir, exp_name)


if __name__ == '__main__':
    ls_vid_fn = [
        'test_4', 
        # 'test_7', 
        # 'test_3', 

        # 'test_1', 
        # 'test_5', 
        # 'test_6', 
        # 'test_2'
    ]

    ls_model_type = [
        # 'added_no_ball_frame',
        # 'not_added',
        # 'added_no_ball_frame_decode_area',
        # 'add_pos_pred_weight',
        'added_no_ball_frame_add_pos_pred_weight_real',
        # 'multiball_added_no_ball_frame_add_pos_pred_weight_real'
    ]

    for game_name in ls_vid_fn:
        for model_type in ls_model_type:
            # if game_name != 'test_6' or model_type != 'added_no_ball_frame_add_pos_pred_weight_real':
            #     continue

            print(f'processing {game_name} with {model_type} model')
            tracker = GameTracker(
                fps = 120, 
                img_w=1920,
                img_h=1080,
                vid_fp=f'samples/{game_name}.mp4',
                vid_res_dir=f'results/{game_name}_{model_type}',
                limit_ball_in_table=False,
                return_frame_with_no_ball=True,
                debug=False,
                # debug_dir=f'full_pipeline_debug/{game_name}_{model_type}_{get_date()}_1',
            )

            ls_cx, ls_cy, fr_indices = tracker.get_list_coord(2211, 2700)
            tracker.plot_ball_pos(ls_cx, ls_cy, fr_indices, save_path='ball_trajectory.png')
            pdb.set_trace()
            

            tracker.generate_hightlight()
            print()
            print('proposed rally: '.capitalize(), tracker.ls_proposed_rally)
            print('final rally: '.capitalize(), tracker.ls_rally)

            print('-------------------------- Inspecting valid rally --------------------------')
            all_infos = []
            for rally in tracker.ls_rally:
                ls_cx, ls_cy, fr_indices = tracker.get_list_coord(rally[0], rally[1])
                print('rally: ', rally)
                print('list extrema x: ', tracker.get_extrema_x(ls_cx, fr_indices)[1])
                info = tracker.get_rally_info(rally[0], rally[1], save_dir=None)
                info['start'] = rally[0]
                info['end']  = rally[1]
                print('info: ', info)
                print('\n')
                all_infos.append(info)

            info_dict = {
                game_name: all_infos
            }
            with open(f'results_pipeline/{game_name}_{model_type}.json', 'w') as f:
                json.dump(info_dict, f)
            print(f'Results saved to results_pipeline/{game_name}_{model_type}.json')

            # ev_data_anno = '/data2/tungtx2/datn/ttnet/dataset/test/annotations/{game_name}/events_markup.json'
            # ev_data = json.load(open(ev_data_anno, 'r'))
            
            # with open('debug_new_model.txt', 'w') as f:
            #     for ev_type in ['bounce', 'net']:
            #         true_indices = [k for k in ev_data.keys() if ev_data[k] == ev_type]
            #         pred_indices = getattr(tracker, f'{ev_type}_indices')
            #         for pred_idx in pred_indices:
            #             nearby_indices = list(range(pred_idx-4, pred_idx+5))
            #             is_matched = False
            #             for idx in nearby_indices:
            #                 if idx in true_indices:
            #                     is_matched = True
            #                     break
                        
            #             if not is_matched:
            #                 f.write(f'false {ev_type} idx: {pred_idx}\n')
            #                 f.write('probs:\n')
            #                 for idx in range(pred_idx-10, pred_idx+10):
            #                     str2write = f'{idx}: {tracker.game_info[idx]["ev_probs"]}\n'
            #                     f.write(str2write)
            #                 f.write('balls:\n')
            #                 for idx in range(pred_idx-10, pred_idx+10):
            #                     str2write = f'{idx}: {tracker.game_info[idx]["ball"]}\n'
            #                     f.write(str2write)
            #                 f.write('\n')

            # pdb.set_trace()

            # game_insights = tracker.save_game_insights()
            # game_insights = {k: v for k, v in sorted(game_insights.items(), key=lambda item: item[0])}
            # pdb.set_trace()

            # tracker.annotate_vid(save_dir=f'full_info_annotated_frames/{game_name}_{model_type}')