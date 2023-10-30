import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import torch
from tqdm.notebook import tqdm
from facenet_pytorch import MTCNN
class Detector:
    def __init__(self, pretrained = 'MTCNN', min_face_size = 50):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if pretrained == 'MTCNN':
            self.detector = MTCNN(min_face_size=min_face_size,device=self.device, post_process=True)
            # self.detector = MTCNN()
    def detect(self, image):
        face = self.detector.detect(image)
        return face

