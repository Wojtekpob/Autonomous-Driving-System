import numpy as np
import cv2
class PathPlanningModule:
    def __init__(self, order=3, image_size=(512, 288)):
        """
        Initializes the path planning module.

        :param order: Degree of the polynomial to fit to the lane points.
        :param image_size: Image size (width, height).
        """
        self.order = order
        self.image_size = image_size
        self.last_detected_lane_lines = None
        self.xm_per_pix = 3.7 / self.image_size[0]
        self.ym_per_pix = 30.0 / self.image_size[1]

    def process_lane_mask(self, lane_mask):
        """
        Processes the lane mask to extract individual lane lines.

        :param lane_mask: Binary lane mask image.
        :return: List of lane lines, each represented as a set of points.
        """
        num_labels, labels_im = cv2.connectedComponents(lane_mask)

        lane_lines = []

        for label in range(1, num_labels):
            component_mask = np.uint8(labels_im == label)
            indices = np.where(component_mask == 1)
            y_vals = indices[0]
            x_vals = indices[1]

            if len(x_vals) == 0:
                continue

            length = y_vals.max() - y_vals.min()

            lane_line = {
                'x_vals': x_vals,
                'y_vals': y_vals,
                'length': length,
                'max_y': y_vals.min()
            }
            lane_lines.append(lane_line)

        return lane_lines

    def select_lane_lines(self, lane_lines):
        """
        Selects up to two most relevant lane lines based on length and extension.

        :param lane_lines: List of lane lines with their properties.
        :return: List of selected lane lines.
        """
        min_length = 50
        valid_lane_lines = [line for line in lane_lines if line['length'] >= min_length]

        if not valid_lane_lines:
            lane_lines.sort(key=lambda x: -x['max_y'])
            selected_lane_lines = [lane_lines[0]] if lane_lines else None
        else:
            valid_lane_lines.sort(key=lambda x: (x['length'], -x['max_y']), reverse=True)
            selected_lane_lines = valid_lane_lines[:2]

        return selected_lane_lines

    def fit_polynomial(self, lane_lines):
        """
        Fits polynomials to the selected lane lines.

        :param lane_lines: List of selected lane lines.
        :return: Combined polynomial coefficients.
        """
        all_x = []
        all_y = []

        for lane_line in lane_lines:
            x_vals = lane_line['x_vals']
            y_vals = lane_line['y_vals']

            all_x.extend(x_vals)
            all_y.extend(y_vals)

        if len(all_x) == 0:
            return None

        coeffs = np.polyfit(all_y, all_x, self.order)

        return coeffs

    def transform_to_vehicle_coordinates(self, x_vals, y_vals):
        """
        Transforms trajectory points from image coordinates to vehicle coordinates.

        :param x_vals: X coordinates in image space.
        :param y_vals: Y coordinates in image space.
        :return: X and Y coordinates in vehicle coordinates.
        """
        y_vehicle = (y_vals - self.image_size[1]) * -self.ym_per_pix
        x_vehicle = (x_vals - self.image_size[0] / 2) * self.xm_per_pix

        return x_vehicle, y_vehicle

    def transform_trajectory_to_image_space(self, x_vehicle, y_vehicle):
        """
        Transforms trajectory points from vehicle coordinates to image space.

        :param x_vehicle: X coordinates in vehicle space.
        :param y_vehicle: Y coordinates in vehicle space.
        :return: X and Y coordinates in image space.
        """
        x_image = x_vehicle / self.xm_per_pix + self.image_size[0] / 2
        y_image = self.image_size[1] - (y_vehicle / self.ym_per_pix)

        return x_image, y_image

    def calculate_errors(self, coeffs):
        """
        Calculates cross-track error (cte) and orientation error (epsi).

        :param coeffs: Polynomial coefficients in vehicle coordinates.
        :return: Cross-track error (cte) and orientation error (epsi).
        """
        cte = np.polyval(coeffs, 0)
        epsi = -np.arctan(coeffs[1])

        return cte, epsi

    def plan_path(self, lane_mask, vehicle_state):
        """
        Main method of the path planning module.

        :param lane_mask: Binary lane mask image.
        :param vehicle_state: Vehicle state (position, orientation, speed).
        :return: Polynomial coefficients, cte, epsi, trajectory in vehicle coordinates.
        """
        lane_lines = self.process_lane_mask(lane_mask)

        if lane_lines is None or len(lane_lines) == 0:
            print("No lane lines detected.")
            return None, None, None, None

        selected_lane_lines = self.select_lane_lines(lane_lines)

        if selected_lane_lines is None:
            print("No suitable lane lines found.")
            return None, None, None, None

        coeffs_image = self.fit_polynomial(selected_lane_lines)

        if coeffs_image is None:
            print("Unable to fit polynomial to lane lines.")
            return None, None, None, None

        y_vals = np.linspace(self.image_size[1] - 1, 0, num=self.image_size[1])

        x_vals = np.polyval(coeffs_image, y_vals)

        x_vehicle, y_vehicle = self.transform_to_vehicle_coordinates(x_vals, y_vals)

        coeffs_vehicle = np.polyfit(x_vehicle, y_vehicle, self.order)

        cte, epsi = self.calculate_errors(coeffs_vehicle)

        self.last_detected_lane_lines = selected_lane_lines

        return coeffs_vehicle, cte, epsi, (x_vehicle, y_vehicle)