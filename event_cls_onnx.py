import onnx, onnxruntime
import pdb
import os
import cv2
import numpy as np
import torch


class EventClsOnnx:
    def __init__(
        self, 
        onnx_path,
        n_input_frames
    ):
        self.ort_session = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        assert onnxruntime.get_device() == 'GPU', 'onnx not running on GPU!'

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


    def forward(self, imgs, pos):
        outputs = self.ort_session.run(
            None,
            {
                'imgs': imgs,
                'pos': pos
            }
        )
        probs = torch.softmax(torch.from_numpy(outputs[0]), dim=1).numpy()
        return probs


    def postprocess(self, probs):
        pred_idx = np.argmax(probs, axis=1)
        return pred_idx


    def predict(self, imgs, pos, return_probs=True):
        probs = self.forward(imgs, pos)
        if return_probs:
            return probs
        
        pred_idx = self.postprocess(probs)
        pred = self.ev_dict[pred_idx[0]]
        return pred
    
    
if __name__ == '__main__':
    onnx_path = 'models/exp4_ce_loss_event_cls_epoch36.onnx'
    predictor = EventClsOnnx(onnx_path)
    imgs = np.random.rand(1, 27, 320, 128).astype(np.float32)
    pos = np.random.rand(1, 9, 2).astype(np.float32)

    output = predictor.forward(imgs, pos)
    print(output.shape)
    pdb.set_trace()