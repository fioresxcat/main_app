
# ffmpeg  \
# -framerate 60 \
# -pattern_type glob \
# -i '/data3/users/tungtx2/main_app/full_info_annotated_frames/test_4_added_no_ball_frame/*.jpg' \
# -c:v libx264 \
# -pix_fmt yuv420p \
# '/data3/users/tungtx2/main_app/full_info_annotated_videos/test_4_added_no_ball_frame.mp4'


# ffmpeg  \
# -framerate 60 \
# -pattern_type glob \
# -i '/data3/users/tungtx2/main_app/full_info_annotated_frames/test_4_not_added/*.jpg' \
# -c:v libx264 \
# -pix_fmt yuv420p \
# '/data3/users/tungtx2/main_app/full_info_annotated_videos/test_4_not_added.mp4'



# ffmpeg  \
# -framerate 60 \
# -pattern_type glob \
# -i '/data3/users/tungtx2/main_app/full_info_annotated_frames/test_7_not_added/*.jpg' \
# -c:v libx264 \
# -pix_fmt yuv420p \
# '/data3/users/tungtx2/main_app/full_info_annotated_videos/test_7_not_added.mp4'



# ffmpeg  \
# -framerate 60 \
# -pattern_type glob \
# -i '/data3/users/tungtx2/main_app/full_info_annotated_frames/test_7_added_no_ball_frame/*.jpg' \
# -c:v libx264 \
# -pix_fmt yuv420p \
# '/data3/users/tungtx2/main_app/full_info_annotated_videos/test_7_added_no_ball_frame.mp4'



# ffmpeg  \
# -framerate 60 \
# -pattern_type glob \
# -i '/data3/users/tungtx2/main_app/full_info_annotated_frames/test_3_not_added/*.jpg' \
# -c:v libx264 \
# -pix_fmt yuv420p \
# '/data3/users/tungtx2/main_app/full_info_annotated_videos/test_3_not_added.mp4'



# ffmpeg  \
# -framerate 60 \
# -pattern_type glob \
# -i '/data3/users/tungtx2/main_app/full_info_annotated_frames/test_3_added_no_ball_frame/*.jpg' \
# -c:v libx264 \
# -pix_fmt yuv420p \
# '/data3/users/tungtx2/main_app/full_info_annotated_videos/test_3_added_no_ball_frame.mp4'



# ffmpeg  \
# -framerate 60 \
# -pattern_type glob \
# -i '/data3/users/tungtx2/main_app/debug/test_4_added_no_ball_frame_decode_area/*.jpg' \
# -c:v libx264 \
# -pix_fmt yuv420p \
# '/data3/users/tungtx2/main_app/full_info_annotated_videos/test_4_added_no_ball_frame_decode_area.mp4'


ffmpeg  \
-framerate 60 \
-pattern_type glob \
-i '/data3/users/tungtx2/main_app/full_info_annotated_frames/test_6_added_no_ball_frame_add_pos_pred_weight_real/*.jpg' \
-c:v libx264 \
-pix_fmt yuv420p \
'/data3/users/tungtx2/main_app/full_info_annotated_videos/test_6_added_no_ball_frame_add_pos_pred_weight_real.mp4'