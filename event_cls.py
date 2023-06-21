from typing import Any, List
from easydict import EasyDict
import pdb
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchmetrics
import pytorch_lightning as pl



class LSTMModel(pl.LightningModule):
    def __init__(self, input_size=2, hidden_size=16, num_layers=2, output_size=16, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
    

class EventClassifierModel(pl.LightningModule):
    def __init__(self, cnn_cfg, lstm_cfg, classifier_dropout, num_classes):
        super().__init__()
        self.cnn_cfg = EasyDict(cnn_cfg)
        self.lstm_cfg = EasyDict(lstm_cfg)

        effb0 = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT)
        effb0.features[0][0] = nn.Conv2d(3*self.cnn_cfg.num_frames, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        effb0.features = effb0.features[:self.cnn_cfg.cut_index]

        self.cnn = nn.Sequential(
            effb0.features,
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(),
        )
        self.lstm = LSTMModel(**lstm_cfg)
        self.fc1 = nn.Linear(120, 32)
        self.act1 = nn.SiLU()
        self.dropout = nn.Dropout(p=classifier_dropout)
        self.fc2 = nn.Linear(32, num_classes)

    
    def forward(self, imgs, pos):
        out_cnn = self.cnn(imgs)    # shape (n x 112)
        out_lstm = self.lstm(pos)   # shape (n x 16)
        fuse = torch.concat([out_cnn, out_lstm], dim=-1)
        x = self.fc1(fuse)
        x = self.act1(x)
        x = self.dropout(x)
        out = self.fc2(x)
        return out


class EventClassifierPredictor:
    def __init__(
        self,
        model: EventClassifierModel,
        device: str
    ):
        self.model = model
        self.device = device
        self.model.eval().to(self.device)
        

    def predict(self, imgs: torch.Tensor, pos: torch.Tensor, return_probs=True) -> Any:
        """
            imgs: shape (n x 27 x 128 x 320), normalized
            pos: shape (n x 9 x 2)
        """
        imgs = imgs.to(self.device)
        pos = pos.to(self.device)
        with torch.no_grad():
            probs = self.model(imgs, pos)
            probs = torch.softmax(probs, dim=-1)
            if return_probs:
                return probs
            pred_idx = torch.argmax(probs, dim=-1)
            return pred_idx


if __name__ == '__main__':
    import pdb
    from easydict import EasyDict
    import yaml

    cnn_cfg = EasyDict(dict(
        num_frames=9
    ))
    lstm_cfg = EasyDict(dict(
        input_size=2, 
        hidden_size=16, 
        num_layers=2, 
        output_size=16
    ))

    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = EasyDict(config)
    model = EventClassifierModel(**config.model.model.init_args)

    pdb.set_trace()
    imgs = torch.rand(2, 27, 128, 128)
    pos = torch.rand(2, 9, 2)
    out = model(imgs, pos)
    pdb.set_trace()
    print(out.shape)