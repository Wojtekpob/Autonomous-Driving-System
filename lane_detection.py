import os
import torch
import cv2
import numpy as np
from model import lanenet
from model.utils import cluster_embed, fit_lanes, sample_from_curve, generate_json_entry, get_color
from torchvision import transforms
class LaneDetectionModule:
    def __init__(self, ckpt_path, arch='enet', dual_decoder=False, device=None):
        """
        Initializes the Lane Detection Module.

        :param ckpt_path: Path to the model checkpoint (.pth file).
        :param arch: Architecture of the network (default 'enet').
        :param dual_decoder: Whether to use dual decoder architecture.
        :param device: Device to run the model on ('cuda' or 'cpu').
        """
        self.ckpt_path = ckpt_path
        self.arch = arch
        self.dual_decoder = dual_decoder
        self.device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mean = np.array([103.939, 116.779, 123.68])
        self.size = (512, 288)
        self._load_model()

    def _load_model(self):
        """
        Loads the neural network model from the checkpoint file.
        """
        if 'fcn' in self.arch.lower():
            arch_name = 'lanenet.LaneNet_FCN_Res'
        elif 'enet' in self.arch.lower():
            arch_name = 'lanenet.LaneNet_ENet'
        elif 'icnet' in self.arch.lower():
            arch_name = 'lanenet.LaneNet_ICNet'
        else:
            raise ValueError(f"Unknown architecture: {self.arch}")

        arch_name += '_1E2D' if self.dual_decoder else '_1E1D'

        self.net = eval(arch_name)()
        self.net = torch.nn.DataParallel(self.net)
        self.net.to(self.device)
        self.net.eval()

        checkpoint = torch.load(self.ckpt_path, map_location=self.device)
        self.net.load_state_dict(checkpoint['model_state_dict'], strict=True)

    def preprocess_image(self, image):
        """
        Preprocesses the input image for the neural network.

        :param image: Input image as a NumPy array (BGR format).
        :return: Preprocessed image tensor.
        """
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32)
        image -= self.mean
        image = np.transpose(image, (2, 0, 1))
        image_tensor = torch.from_numpy(image).float() / 255.0
        return image_tensor

    def predict(self, image):
        """
        Performs lane detection on the input image.

        :param image: Input image as a NumPy array (BGR format).
        :return: Predicted lane mask as a NumPy array.
        """
        input_tensor = self.preprocess_image(image)
        input_batch = input_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            embeddings, logit = self.net(input_batch)
            pred_bin_batch = torch.argmax(logit, dim=1, keepdim=True)

        pred_mask = pred_bin_batch[0].cpu().numpy().transpose(1, 2, 0)
        pred_mask = pred_mask.astype(np.uint8) * 255
        return pred_mask