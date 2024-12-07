import cv2
import numpy as np

class VisualizationModule:
    def __init__(self, image_size=(512, 288)):
        """
        Initializes the Visualization Module.

        :param image_size: Tuple of (width, height) representing the image size.
        """
        self.image_size = image_size

    def visualize(self, image, lane_mask, trajectory_coeffs=None, lane_lines=None, y_vals_small=None, show=False, save_path=None):
        """
        Visualizes the lane detection and planned trajectory on the input image.

        :param image: Input image as a NumPy array (BGR format).
        :param lane_mask: Lane mask as a NumPy array (same size as image).
        :param trajectory_coeffs: Polynomial coefficients of the trajectory.
        :param lane_lines: List of polynomial coefficients for lane lines.
        :param y_vals_small: Array of y-values in resized image coordinates.
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

        h_small, w_small = self.image_size[1], self.image_size[0]
        x_scale = image.shape[1] / w_small
        y_scale = image.shape[0] / h_small

        if y_vals_small is None:
            max_y = h_small - 1 
            min_y = int(h_small * 0.5) 
            y_vals_small = np.linspace(max_y, min_y, num=100)

        y_vals = y_vals_small * y_scale

        if lane_lines is not None:
            for param in lane_lines:
                x_vals_small = np.polyval(param, y_vals_small)
                x_vals = x_vals_small * x_scale
                points = np.vstack((x_vals, y_vals)).T
                points[:, 0] = np.clip(points[:, 0], 0, image.shape[1] - 1)
                points[:, 1] = np.clip(points[:, 1], 0, image.shape[0] - 1)
                points = points.astype(np.int32)
                cv2.polylines(combined_img, [points], isClosed=False, color=(0, 255, 0), thickness=5)

        if trajectory_coeffs is not None:
            x_vals_small = np.polyval(trajectory_coeffs, y_vals_small)
            x_vals = x_vals_small * x_scale
            points = np.vstack((x_vals, y_vals)).T
            points[:, 0] = np.clip(points[:, 0], 0, image.shape[1] - 1)
            points[:, 1] = np.clip(points[:, 1], 0, image.shape[0] - 1)
            points = points.astype(np.int32)
            cv2.polylines(combined_img, [points], isClosed=False, color=(0, 0, 255), thickness=5)

        center_x = image.shape[1] // 2
        bottom_y = image.shape[0] - 10 
        offset = 20 

        pt1 = (center_x - offset, bottom_y - offset)  
        pt2 = (center_x + offset, bottom_y + offset) 
        pt3 = (center_x - offset, bottom_y + offset) 
        pt4 = (center_x + offset, bottom_y - offset)  

        cv2.line(combined_img, pt1, pt2, (255, 0, 0), 5)  
        cv2.line(combined_img, pt3, pt4, (255, 0, 0), 5) 


        if show:
            cv2.imshow('Visualization', combined_img)
            cv2.waitKey(1)

        if save_path is not None:
            cv2.imwrite(save_path, combined_img)

        return combined_img