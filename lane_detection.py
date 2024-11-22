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
        Initializes lane detection module

        :param ckpt_path: Path to model.
        :param arch: Network architecture.
        :param dual_decoder: Whether to use dual decoder.
        :param device: 'cuda:0' or 'cpu'.
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
        Loads model from checkpoint file.
        """
        if 'fcn' in self.arch.lower():
            arch_name = 'lanenet.LaneNet_FCN_Res'
        elif 'enet' in self.arch.lower():
            arch_name = 'lanenet.LaneNet_ENet'
        elif 'icnet' in self.arch.lower():
            arch_name = 'lanenet.LaneNet_ICNet'
        else:
            raise ValueError(f"Unkown architecture: {self.arch}")

        arch_name += '_1E2D' if self.dual_decoder else '_1E1D'

        self.net = eval(arch_name)()
        self.net = torch.nn.DataParallel(self.net)
        self.net.to(self.device)
        self.net.eval()

        checkpoint = torch.load(self.ckpt_path, map_location=self.device)
        self.net.load_state_dict(checkpoint['model_state_dict'], strict=True)

    def preprocess_image(self, image):
        """
        Preprocesses input image.

        :param image: Image in numpy array format (BGR).
        :return: Image tensor, that is ready for network.
        """
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32)
        image -= self.mean
        image = np.transpose(image, (2, 0, 1))
        image_tensor = torch.from_numpy(image).float() / 255.0
        return image_tensor

    def predict(self, image):
        """
        Predicts on single image.

        :param image: Image in numpy array format (BGR).
        :return: Prediction result (eg. segmentation mask).
        """
        input_tensor = self.preprocess_image(image)
        input_batch = input_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            embeddings, logit = self.net(input_batch)
            pred_bin_batch = torch.argmax(logit, dim=1, keepdim=True)

        pred_mask = pred_bin_batch[0].cpu().numpy().transpose(1, 2, 0)
        pred_mask = pred_mask.astype(np.uint8) * 255  
        
        return pred_mask

    def visualize_prediction(self, image, pred_mask, show=False, save_path=None):
        """
        Visualizes prediction result on input image.

        :param image: Input image in numpy array format (BGR).
        :param pred_mask: Prediction mask (numpy array).
        :param show: Whether to show result on screen.
        :param save_path: Path to save result (if None, then doesn't save).
        """
        input_rgb = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
        input_rgb = input_rgb.astype(np.uint8)

        pred_mask_rgb = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)

        combined_img = cv2.addWeighted(input_rgb, 1.0, pred_mask_rgb, 0.5, 0)

        if show:
            cv2.imshow('Prediction', combined_img)
            cv2.waitKey(1) 

        if save_path is not None:
            cv2.imwrite(save_path, combined_img)

        return combined_img
