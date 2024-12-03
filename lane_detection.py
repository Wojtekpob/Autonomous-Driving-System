import os
import torch
import cv2
import numpy as np
from model import lanenet
from model.utils import cluster_embed, fit_lanes, sample_from_curve, get_color
from torchvision import transforms
import warnings

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
        :return: Tuple of (predicted lane mask, selected polynomial coefficients).
        """
        input_tensor = self.preprocess_image(image)
        input_batch = input_tensor.unsqueeze(0).to(self.device)
        h, w = self.size[1], self.size[0]

        with torch.no_grad():
            embeddings, logit = self.net(input_batch)
            pred_bin_batch = torch.argmax(logit, dim=1, keepdim=True)

        pred_bin = pred_bin_batch[0, 0].cpu().numpy().astype(np.uint8)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(pred_bin, connectivity=8)

        center_x = w / 2
        min_area = 500  
        lanes_info = []

        for label in range(1, num_labels): 
            mask = labels == label
            area = stats[label, cv2.CC_STAT_AREA]
            if area < min_area:
                continue

            y_coords, x_coords = np.where(mask)

            max_y = int(h * 0.9) 
            valid_indices = y_coords >= int(h * 0.5) 
            y_coords = y_coords[valid_indices]
            x_coords = x_coords[valid_indices]

            if len(y_coords) < 2:
                continue

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', np.RankWarning)
                try:
                    coeffs = np.polyfit(y_coords, x_coords, deg=2)
                except np.RankWarning:
                    continue

            y_bottom = h - 1
            x_bottom = np.polyval(coeffs, y_bottom)
            distance_from_center = x_bottom - center_x

            lanes_info.append((distance_from_center, coeffs))

        left_lanes = [info for info in lanes_info if info[0] < 0]
        right_lanes = [info for info in lanes_info if info[0] >= 0]

        if left_lanes:
            left_lane = max(left_lanes, key=lambda x: x[0])  # Closest to center
        else:
            left_lane = None

        if right_lanes:
            right_lane = min(right_lanes, key=lambda x: x[0])  # Closest to center
        else:
            right_lane = None

        selected_params = []
        if left_lane:
            selected_params.append(left_lane[1])
        if right_lane:
            selected_params.append(right_lane[1])

        pred_mask = pred_bin[:, :, np.newaxis] * 255

        return pred_mask, selected_params