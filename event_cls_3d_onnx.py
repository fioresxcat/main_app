import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import onnx, onnxruntime
import pdb
import os
import cv2
import numpy as np
import torch
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)


class EventCls3DOnnx:
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
            0: 'bounce',
            1: 'net',
            2: 'empty'
        }
        self.n_input_frames = n_input_frames
        self.normalize_video = NormalizeVideo(
            mean = [0.45, 0.45, 0.45],
            std = [0.225, 0.225, 0.225]
        )
        self.crop_size = (182, 182)


    def forward(self, imgs):
        outputs = self.ort_session.run(
            None,
            {
                'imgs': imgs,
            }
        )
        probs = torch.softmax(torch.from_numpy(outputs[0]), dim=1).numpy()
        return probs


    def preprocess(self, imgs):
        """
            list of frames (np.array shape 128 x 320 x 3)
        """
        imgs = [cv2.resize(img, self.crop_size) for img in imgs]
        imgs = np.stack(imgs, axis=0)
        imgs = imgs.transpose(3, 0, 1, 2)    # shape 3 x n_frames x 182 x 182
        imgs = imgs / 255.
        imgs = self.normalize_video(torch.from_numpy(imgs)).numpy()
        imgs = np.expand_dims(imgs, axis=0).astype(np.float32)
        return imgs


    def postprocess(self, probs):
        pred_indices = np.argmax(probs, axis=1)
        return pred_indices


    def predict(self, imgs, return_probs=True):
        inp = self.preprocess(imgs)
        probs = self.forward(inp)
        if return_probs:
            return probs
        
        pred_indices = self.postprocess(probs)
        return pred_indices
    
    
if __name__ == '__main__':
    onnx_path = 'models/epoch73_event_cls_3d_remake_label.onnx'
    predictor = EventCls3DOnnx(onnx_path, n_input_frames=15)
    imgs = torch.randn(2, 3, 9, 182, 182, dtype=torch.float32).numpy()
    output = predictor.predict(imgs, return_probs=False)
    print(output.shape)
    pdb.set_trace()