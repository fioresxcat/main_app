import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import onnx, onnxruntime
import pdb
import os
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
import cv2
import numpy as np
import torch


class ServeDetectOnnx:
    def __init__(
        self, 
        onnx_path,
        n_input_frames
    ):
        # self.ort_session = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.ort_session = onnxruntime.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        # assert onnxruntime.get_device() == 'GPU', 'onnx not running on GPU!'

        print('------------- ONNX model summary ------------')
        for input in self.ort_session.get_inputs():
            print(input.name, '-', input.type, '-', input.shape)
        print()

        self.ev_dict = {
            0: 'no_serve',
            1: 'serve',
        }
        self.n_input_frames = n_input_frames
        self.normalize_video = NormalizeVideo(
            mean = [0.45, 0.45, 0.45],
            std = [0.225, 0.225, 0.225]
        )
        self.crop_size = (256, 256)


    def preprocess(self, frame_list):
        frames = [cv2.resize(frame, self.crop_size) for frame in frame_list]
        frames = np.stack(frames, axis=0)   # shape n_frames x 256 x 256 x 3
        frames = frames.transpose(3, 0, 1, 2)   # shape n_frames x 3 x 256 x 256
        frames = frames/255.
        frames = self.normalize_video(torch.from_numpy(frames)).numpy().astype(np.float32)
        frames = np.expand_dims(frames, axis=0)
        return frames


    def forward(self, imgs):
        outputs = self.ort_session.run(
            None,
            {
                'imgs': imgs,
            }
        )
        probs = torch.sigmoid(torch.from_numpy(outputs[0])).numpy()
        return probs


    def postprocess(self, probs):
        pred_indices = (probs > 0.5).astype(np.int32)
        return pred_indices


    def predict(self, imgs, return_probs=False):
        inp = self.preprocess(imgs)
        probs = self.forward(inp)
        if return_probs:
            return probs
        
        pred_indices = self.postprocess(probs)
        return pred_indices
    
    
if __name__ == '__main__':
    onnx_path = 'models/serve_detect_epoch0.onnx'
    predictor = ServeDetectOnnx(onnx_path, n_input_frames=15)
    imgs = torch.randn(2, 3, 15, 182, 182, dtype=torch.float32).numpy()
    output = predictor.predict(imgs, return_probs=True)
    print(output.shape)
    pdb.set_trace()