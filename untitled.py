import cv2
from ultralytics import YOLO

def add_frame_number(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = 720
    height = 480
    out = cv2.VideoWriter(output_path, codec, fps, (width, height))
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        out.write(frame)
        print(f'done {frame_count} frames')

        if frame_count == 100:
            break
    cap.release()
    out.release()


if __name__ == '__main__':
    # vid_fp = '/data2/tungtx2/datn/samples/test_2.mp4'
    # out_fp = '/data2/tungtx2/datn/samples/test_2_numbered.mp4'
    # add_frame_number(vid_fp, out_fp)

    model = YOLO('models/yolo-segment-train2-best.pt')
    model.export(
        format='engine',
        imgsz=640,
        # optimize=True,
        dynamic=True,
        simplify=True,
        # opset=14,
        # half=True
    )
