import onnx, onnxruntime
import pdb
import os
import cv2
import numpy as np


def decode_hm(hm, conf_thresh):
    """
        hm shape: n x 1 x out_w x out_h
        if there is a heatmap in the batch with no detected ball, will return ball_pos as [0, 0]
    """
    hm = hm.squeeze(axis=1)
    hm = np.where(hm < conf_thresh, 0, hm)
    max_indices = np.argmax(hm.reshape(hm.shape[0], -1), axis=1)
    pos = np.vstack([max_indices // hm.shape[1], max_indices % hm.shape[1]]).T
    pos[:, [0, 1]] = pos[:, [1, 0]]
    return pos


def decode_hm_by_contour(batch_hm, conf_thresh):
    """
        hm shape: n x 1 x out_w x out_h
        om shape: n x 2 x out_w x out_h
        if there is a heatmap in the batch with no detected ball, will return ball_pos as [0, 0]

    """
    batch_hm = batch_hm.squeeze(dim=1).cpu().numpy()
    batch_hm_int = (batch_hm*255).astype(np.uint8)
    batch_pos = []
    for idx, hm_int in enumerate(batch_hm_int):
        hm = batch_hm[idx]
        ret, binary_hm = cv2.threshold(hm_int, conf_thresh*255, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary_hm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            xmin, ymin, w, h = cv2.boundingRect(largest_contour)
            max_idx = np.unravel_index(np.argmax(hm[ymin:ymin+h, xmin:xmin+w]), (h, w))
            ball_x, ball_y = (max_idx[1]+xmin, max_idx[0] + ymin)
            batch_pos.append(np.array([ball_x, ball_y]))
        else:
            batch_pos.append(np.array([0, 0]))
    
    batch_pos = np.stack(batch_pos, axis=0)
    return batch_pos



class CenternetOnnx:
    def __init__(
        self, 
        onnx_path,
        n_input_frames,
        decode_by_area=False,
        conf_thresh=0.5
    ):  
    
        self.ort_session = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        # self.ort_session = onnxruntime.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

        assert onnxruntime.get_device() == 'GPU', 'onnx not running on GPU!'

        print('------------- ONNX model summary ------------')
        for input in self.ort_session.get_inputs():
            print(input.name, '-', input.type, '-', input.shape)
        print()

        self.in_h, self.in_w = input.shape[2:]
        self.n_input_frames = n_input_frames
        self.decode_by_area = decode_by_area
        self.conf_thresh = conf_thresh


    def forward(self, imgs):
        outputs = self.ort_session.run(
            None,
            {
                'imgs': imgs
            }
        )
        hm, om = outputs
        return hm, om


    def postprocess(
        self, 
        hm, 
        om, 
    ):
        out_h, out_w = hm.shape[2:]

        if self.decode_by_area:
            batch_pos = decode_hm_by_contour(hm, self.conf_thresh)
        else:
            batch_pos = decode_hm(hm, self.conf_thresh)    # shape nx2

        max_values = np.max(hm.squeeze(axis=1).reshape(hm.shape[0], -1), axis=1)
        max_values = max_values.reshape(-1, 1)

        batch_final_pos = []
        for i in range(batch_pos.shape[0]):
            pos = batch_pos[i]
            if np.all(pos == 0):
                offset = np.array([0, 0])
            else:
                offset = om[i][:, pos[1], pos[0]]
            final_pos = pos + offset
            final_pos = final_pos / np.array([out_w, out_h]) * np.array([self.in_w, self.in_h])
            batch_final_pos.append(final_pos)
        batch_final_pos = np.stack(batch_final_pos, axis=0).astype(np.int)

        return batch_final_pos, max_values


    def predict(self, imgs):
        """ 
            imgs: shape n x 3 x 512 x 512, normalized
        """
        hm, om = self.forward(imgs)
        batch_final_pos, _ = self.postprocess(hm, om)
        return batch_final_pos / np.array([self.in_w, self.in_h])


if __name__ == '__main__':
    onnx_path = 'models/exp38_centernet_ason_fixed_mask_ball_epoch_18.onnx'
    predictor = CenternetOnnx(onnx_path)
    imgs = np.random.rand(3, 15, 512, 512).astype(np.float32)
    hm, om = predictor.forward(imgs)
    pos, probs = predictor.postprocess(hm, om)
    print(pos)
    print(probs)
    print(hm.shape)
    print(om.shape)
    pdb.set_trace() 