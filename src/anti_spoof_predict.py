import os
import cv2
import math
import torch
import numpy as np
import torch.nn.functional as F


from src.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2,MiniFASNetV1SE,MiniFASNetV2SE
from src.data_io import transform as trans
from src.utility import get_kernel, parse_model_name

MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE':MiniFASNetV1SE,
    'MiniFASNetV2SE':MiniFASNetV2SE
}

class AntiSpoof():
    def __init__(self, model_path =r'weights\anti_spoof_models\4_0_0_80x80_MiniFASNetV1SE.pth' ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.load_model(model_path)
    def load_model(self, model_path):
        # define model
        model_name = os.path.basename(model_path)
        h_input, w_input , model_type, _ = parse_model_name(model_name)
        self.h_input, self.w_input = h_input, w_input 
        self.kernel_size = get_kernel(h_input, w_input,)
        self.model = MODEL_MAPPING[model_type](conv6_kernel=self.kernel_size).to(self.device)

        # load model weight
        state_dict = torch.load(model_path, map_location=self.device)
        keys = iter(state_dict)
        first_layer_name = keys.__next__()
        if first_layer_name.find('module.') >= 0:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                name_key = key[7:]
                new_state_dict[name_key] = value
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(state_dict)
        return None

    def predict(self, img):
        img = cv2.resize(img, (self.h_input,self.w_input))
        test_transform = trans.Compose([
            trans.ToTensor(),
        ])
        img = test_transform(img)
        img = img.unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            result = self.model.forward(img)
            result = F.softmax(result).cpu().numpy()
        print(result)
        return result











