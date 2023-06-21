# ffmpeg  \
# -framerate 30 \
# -pattern_type glob \
# -i '/data2/tungtx2/datn/model_output/test_2_regen/results/*.jpg' \
# -c:v libx264 \
# -pix_fmt yuv420p \
# '/data2/tungtx2/datn/model_output/test_2_regen/test2_annotated.mp4'


# ffmpeg  \
# -framerate 30 \
# -pattern_type glob \
# -i '/data2/tungtx2/datn/model_output/test_7_regen/results/*.jpg' \
# -c:v libx264 \
# -pix_fmt yuv420p \
# '/data2/tungtx2/datn/model_output/test_7_regen/test7_annotated.mp4'

ffmpeg  \
-framerate 30 \
-pattern_type glob \
-i '/data2/tungtx2/datn/main_app/results/test_4_annotate_main/*.jpg' \
-c:v libx264 \
-pix_fmt yuv420p \
'/data2/tungtx2/datn/main_app/results/test4_annotated_main_result.mp4'

# ffmpeg  \
# -framerate 30 \
# -pattern_type glob \
# -i '/data2/tungtx2/datn/model_output/test_4_regen/results/*.jpg' \
# -c:v libx264 \
# -pix_fmt yuv420p \
# '/data2/tungtx2/datn/model_output/test_4_regen/test4_annotated.mp4'