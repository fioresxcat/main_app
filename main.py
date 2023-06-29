import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from my_utils import *
from ball_tracker import *


def get_info(game_info, fr_idx, info_type):
    for idx in range(fr_idx-10, fr_idx+10):
        print(idx, game_info[idx][info_type])

        
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
        ls_cx, ls_cy, fr_indices = tracker.get_list_coord(rally[0], rally[1])
        print('rally: ', rally)
        print('list extrema x: ', tracker.get_extrema_x(ls_cx, fr_indices)[1])
        info = tracker.get_rally_info(rally[0], rally[1], save_dir='debug/test_7_rally_info')
        print('info: ', info)
        info['start'] = rally[0]
        info['end']  = rally[1]
        print('\n')

    ev_data_anno = '/data/tungtx2/datn/dataset/test/annotations/test_4/events_markup.json'
    ev_data = json.load(open(ev_data_anno, 'r'))
    
    with open('debug_new_model.txt', 'w') as f:
        for ev_type in ['bounce', 'net']:
            true_indices = [k for k in ev_data.keys() if ev_data[k] == ev_type]
            pred_indices = getattr(tracker, f'{ev_type}_indices')
            for pred_idx in pred_indices:
                nearby_indices = list(range(pred_idx-4, pred_idx+5))
                is_matched = False
                for idx in nearby_indices:
                    if idx in true_indices:
                        is_matched = True
                        break
                
                if not is_matched:
                    f.write(f'false {ev_type} idx: {pred_idx}\n')
                    f.write('probs:\n')
                    for idx in range(pred_idx-10, pred_idx+10):
                        str2write = f'{idx}: {tracker.game_info[idx]["ev_probs"]}\n'
                        f.write(str2write)
                    f.write('balls:\n')
                    for idx in range(pred_idx-10, pred_idx+10):
                        str2write = f'{idx}: {tracker.game_info[idx]["ball"]}\n'
                        f.write(str2write)
                    f.write('\n')

    pdb.set_trace()
    # gi = tracker.save_game_insights()
    # # sort gi by keys
    # gi = {k: v for k, v in sorted(gi.items(), key=lambda item: item[0])}
    # pdb.set_trace()

    # tracker.annotate_vid(save_dir='results/test_4_annotate_main')