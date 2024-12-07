import os
import torch
import cv2
import numpy as np
from model import lanenet
from model.utils import get_color
from torchvision import transforms
import warnings
from collections import deque

class LaneDetectionModule:
    def __init__(self, ckpt_path, arch='enet', dual_decoder=False, device=None, history_length=5):
        """
        Initializes the Lane Detection Module.

        :param ckpt_path: Path to the model checkpoint (.pth file).
        :param arch: Architecture of the network (default 'enet').
        :param dual_decoder: Whether to use dual decoder architecture.
        :param device: Device to run the model on ('cuda' or 'cpu').
        :param history_length: Number of past predictions to keep for smoothing.
        """
        self.ckpt_path = ckpt_path
        self.arch = arch
        self.dual_decoder = dual_decoder
        self.device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mean = np.array([103.939, 116.779, 123.68])
        self.size = (512, 288)
        self.history_length = history_length
        self.left_lane_history = deque(maxlen=self.history_length)
        self.right_lane_history = deque(maxlen=self.history_length)
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

    def pad_polynomial(self, coeffs, target_degree=2):
        """
        Pads the polynomial coefficients with zeros to match the target degree.

        :param coeffs: Polynomial coefficients (highest degree first).
        :param target_degree: The degree to pad to.
        :return: Padded polynomial coefficients.
        """
        current_degree = len(coeffs) - 1
        if current_degree < target_degree:
            padding = np.zeros(target_degree - current_degree)
            padded_coeffs = np.concatenate((padding, coeffs))
            return padded_coeffs
        return coeffs

    def average_polynomials(self, polynomials):
        """
        Averages a list of polynomials.

        :param polynomials: List of polynomial coefficients arrays.
        :return: Averaged polynomial coefficients.
        """
        if not polynomials:
            return None
        max_degree = max(len(p) for p in polynomials) - 1
        padded_polys = [self.pad_polynomial(p, target_degree=max_degree) for p in polynomials]
        avg_poly = np.mean(padded_polys, axis=0)
        return avg_poly

    def predict(self, image):
        """
        Performs lane detection on the input image, fits polynomials to the two lanes closest to the vehicle center,
        and smooths them using historical data.

        :param image: Input image as a NumPy array (BGR format).
        :return: Tuple of (predicted lane mask visualization, selected polynomial coefficients).
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
            mask = (labels == label)
            area = stats[label, cv2.CC_STAT_AREA]
            if area < min_area:
                continue

            y_coords, x_coords = np.where(mask)
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

        lanes_info.sort(key=lambda x: abs(x[0]))

        left_lane = None
        right_lane = None
        for dist, coeffs in lanes_info:
            if dist < 0 and left_lane is None:
                left_lane = coeffs
            elif dist >= 0 and right_lane is None:
                right_lane = coeffs
            if left_lane is not None and right_lane is not None:
                break

        selected_params = []
        if left_lane is not None:
            self.left_lane_history.append(left_lane)
            averaged_left = self.average_polynomials(list(self.left_lane_history))
            selected_params.append(averaged_left)
        if right_lane is not None:
            self.right_lane_history.append(right_lane)
            averaged_right = self.average_polynomials(list(self.right_lane_history))
            selected_params.append(averaged_right)

        pred_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for idx, coeffs in enumerate(selected_params):
            y_vals = np.linspace(0, h - 1, num=h)
            x_vals = np.polyval(coeffs, y_vals)
            pts = np.vstack((x_vals, y_vals)).T
            pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < w)]
            pts = pts.astype(np.int32)
            pts = pts.reshape((-1, 1, 2))
            color = get_color(idx)
            cv2.polylines(pred_mask, [pts], False, color, thickness=2)

        return pred_mask, selected_params
