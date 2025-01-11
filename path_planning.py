import numpy as np

class PathPlanningModule:
    def __init__(self, order=3, image_size=(512, 288)):
        """
        Initializes the PathPlanningModule.

        Args:
            order (int, optional): Base degree of the polynomial to fit to the trajectory.
                                   Defaults to 3.
            image_size (tuple, optional): Tuple representing (width, height) of the image.
                                          Defaults to (512, 288).
        """
        self.order = order
        self.image_size = image_size

    def transform_to_vehicle_coords(self, image_coeffs):
        """
        Transforms a second-degree polynomial from image coordinates to vehicle coordinates.

        The polynomial in image coordinates is defined as:
            x_img(y_img) = a2 * y_img² + a1 * y_img + a0

        **Image Coordinate System:**
            - Origin: (0, 0) at the top-left corner.
            - X-axis: Points to the right.
            - Y-axis: Points downward.

        **Vehicle Coordinate System:**
            - Origin: Corresponds to (x_img = w/2, y_img = h - 1) in image coordinates,
              where w, h = self.image_size.
            - X_v-axis: Points upward (x_v = (h - 1) - y_img).
            - Y_v-axis: Points to the right (y_v = x_img - (w / 2)).

        Args:
            image_coeffs (array-like): Coefficients of the polynomial in image coordinates,
                                       e.g. [a2, a1, a0].

        Returns:
            numpy.ndarray: Coefficients of the polynomial in vehicle coordinates,
                           ordered as [b2, b1, b0].
        """
        if len(image_coeffs) < 3:
            image_coeffs = np.pad(image_coeffs, (3 - len(image_coeffs), 0), 'constant')

        a2, a1, a0 = image_coeffs[:3]

        w, h = self.image_size
        b2 = a2
        b1 = -(a1 + 2 * (h - 1) * a2)
        b0 = a0 + a1 * (h - 1) + a2 * ((h - 1) ** 2) - (w / 2)

        return np.array([b2, b1, b0])

    def calculate_errors(self, car_coeffs):
        """
        Calculates the Cross-Track Error (CTE) and Orientation Error (EPSI) for a
        second-degree polynomial in vehicle coordinates.

        Given the polynomial in vehicle coordinates:
            y_v = b2 * x_v² + b1 * x_v + b0

        At x_v = 0:
            - CTE (Cross-Track Error) is y_v(0) = b0
            - EPSI (Orientation Error) = -(arctan(b1))

        Args:
            car_coeffs (array-like): Coefficients [b2, b1, b0].

        Returns:
            tuple:
                float: Cross-Track Error (cte).
                float: Orientation Error (epsi).
        """
        b2, b1, b0 = car_coeffs[:3]

        cte = b0
        psides = np.arctan(b1)
        epsi = -psides

        return cte, epsi

    def plan_path(self, lane_polynomials, vehicle_state):
        """
        Plans a path by fitting a second-degree polynomial in image coordinates,
        transforming it to vehicle coordinates, and computing CTE and EPSI.

        Steps:
            1. Sample points between the lane polynomials.
            2. Fit a second-degree polynomial in image coords: x_img(y_img) = a2*y_img² + a1*y_img + a0.
            3. Transform to vehicle coords.
            4. Compute cte and epsi from the resulting polynomial.

        Args:
            lane_polynomials (list of array-like): Polynomial coeffs for lane lines
                                                   [a2, a1, a0].
            vehicle_state (array-like): Current vehicle state, e.g. [x, y, psi, v, cte, epsi].

        Returns:
            tuple:
                numpy.ndarray or None: Coeffs of the fitted polynomial in image coords,
                                       e.g. [a2, a1, a0].
                float or None: Cross-Track Error (cte).
                float or None: Orientation Error (epsi).
                numpy.ndarray or None: Coefficients in vehicle coords [b2, b1, b0].
        """
        if lane_polynomials is None or len(lane_polynomials) == 0:
            print("No lane polynomials provided.")
            return None, None, None, None

        w, h = self.image_size

        if len(lane_polynomials) == 1:
            lane_poly_left = lane_polynomials[0]
            lane_poly_right = lane_polynomials[0]
        else:
            lane_poly_left = lane_polynomials[0]
            lane_poly_right = lane_polynomials[1]

        num_end_points = 500
        num_samples = 10

        y_vals = np.concatenate([
            np.linspace(0, h - 1, num_samples),
            np.full(num_end_points, h - 1)
        ])

        x_mid_points = []
        for y in y_vals:
            x_left = np.polyval(lane_poly_left, y)
            x_right = np.polyval(lane_poly_right, y)
            x_mid = (x_left + x_right) / 2.0
            x_mid_points.append(x_mid)
        x_mid_points = np.array(x_mid_points)

        for i in range(-num_end_points, 0):
            x_mid_points[i] = w / 2

        coeffs = np.polyfit(y_vals, x_mid_points, 2)

        car_coeffs = self.transform_to_vehicle_coords(coeffs)

        cte, epsi = self.calculate_errors(car_coeffs)

        return coeffs, cte, epsi, car_coeffs
