import numpy as np

class PathPlanningModule:
    def __init__(self, order=3, image_size=(512, 288)):
        """
        Initializes the path planning module.

        :param order: Degree of the polynomial to fit to the trajectory.
        :param image_size: Image size (width, height).
        """
        self.order = order
        self.image_size = image_size

    def calculate_errors(self, coeffs):
        """
        Calculates cross-track error (cte) and orientation error (epsi).

        :param coeffs: Polynomial coefficients in image coordinates.
        :return: Cross-track error (cte) and orientation error (epsi).
        """
        x_vehicle = self.image_size[0] / 2 
        y_vehicle = self.image_size[1] - 1

        x_desired = np.polyval(coeffs, y_vehicle)

        cte = x_desired - x_vehicle

        derivative = np.polyder(coeffs)
        theta_desired = np.arctan(np.polyval(derivative, y_vehicle))
        theta_vehicle = 0 
        epsi = theta_vehicle - theta_desired

        return cte, epsi

    def generate_trajectory_coeffs(self, lane_polynomials):
        """
        Generates a trajectory polynomial based on the predicted lane polynomials.

        :param lane_polynomials: List of polynomial coefficients (left and right lanes).
        :return: Polynomial coefficients of the trajectory.
        """
        if len(lane_polynomials) == 2:
            coeffs_left = lane_polynomials[0]
            coeffs_right = lane_polynomials[1]

            if len(coeffs_left) != len(coeffs_right):
                max_len = max(len(coeffs_left), len(coeffs_right))
                coeffs_left = np.pad(coeffs_left, (max_len - len(coeffs_left), 0), 'constant')
                coeffs_right = np.pad(coeffs_right, (max_len - len(coeffs_right), 0), 'constant')

            coeffs = (coeffs_left + coeffs_right) / 2
        else:
            coeffs = lane_polynomials[0]

        return coeffs

    def plan_path(self, lane_polynomials, vehicle_state):
        """
        Main method of the path planning module.

        :param lane_polynomials: List of polynomial coefficients from LaneDetectionModule.
        :param vehicle_state: Vehicle state (position, orientation, speed).
        :return: Polynomial coefficients in image coordinates, cte, epsi.
        """
        if lane_polynomials is None or len(lane_polynomials) == 0:
            print("No lane polynomials provided.")
            return None, None, None

        coeffs = self.generate_trajectory_coeffs(lane_polynomials)

        cte, epsi = self.calculate_errors(coeffs)

        return coeffs, cte, epsi