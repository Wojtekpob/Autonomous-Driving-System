import cv2
import numpy as np

class VisualizationModule:
    def __init__(self, image_size=(512, 288)):
        """
        Initializes the Visualization Module.

        :param image_size: Tuple of (width, height) representing the image size.
        """
        self.image_size = image_size

    def visualize(self, image, lane_mask, trajectory=None, lane_lines=None, show=False, save_path=None):
        """
        Visualizes the lane detection and planned trajectory on the input image.

        :param image: Input image as a NumPy array (BGR format).
        :param lane_mask: Lane mask as a NumPy array (same size as image).
        :param trajectory: Tuple of (x_vals, y_vals) for the trajectory in image space.
        :param lane_lines: List of lane lines to draw.
        :param show: If True, displays the image in a window.
        :param save_path: If provided, saves the visualized image to the specified path.
        :return: Image with visualizations overlaid.
        """
        if len(lane_mask.shape) == 2 or lane_mask.shape[2] == 1:
            lane_mask_rgb = cv2.cvtColor(lane_mask, cv2.COLOR_GRAY2BGR)
        else:
            lane_mask_rgb = lane_mask

        if lane_mask_rgb.shape != image.shape:
            lane_mask_rgb = cv2.resize(lane_mask_rgb, (image.shape[1], image.shape[0]))

        combined_img = cv2.addWeighted(image, 1.0, lane_mask_rgb, 0.5, 0)

        if trajectory is not None:
            x_vals, y_vals = trajectory
            x_vals = np.clip(x_vals, 0, image.shape[1] - 1).astype(np.int32)
            y_vals = np.clip(y_vals, 0, image.shape[0] - 1).astype(np.int32)
            for x, y in zip(x_vals, y_vals):
                cv2.circle(combined_img, (x, y), radius=2, color=(0, 0, 255), thickness=-1)  # Red dots

        if show:
            cv2.imshow('Visualization', combined_img)
            cv2.waitKey(1)

        if save_path is not None:
            cv2.imwrite(save_path, combined_img)

        return combined_img