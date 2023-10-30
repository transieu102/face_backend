from facenet_pytorch import InceptionResnetV1
import torch
from PIL import Image
import torchvision.transforms as transforms
class Extractor:
    def __init__(self):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.extractor = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.preprocess = transforms.Compose([
                            transforms.ToPILImage(),       # Convert the NumPy array to a PIL Image
                            transforms.Resize((160, 160)),  # Resize the image to the expected size
                            transforms.ToTensor(),         # Convert the image to a tensor
                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the image
                        ])
        self.vector_shape = {
            'vggface2' : [1,512]
        }
    def extract(self, image):
        vector = self.extractor(self.preprocess(image).unsqueeze(0))
        return vector