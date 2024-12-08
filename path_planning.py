import numpy as np
from math import comb

class PathPlanningModule:
    def __init__(self, order=3, image_size=(512, 288)):
        """
        Initializes the path planning module.

        :param order: Base degree of the polynomial to fit to the trajectory.
        :param image_size: (width, height)
        """
        self.order = order
        self.image_size = image_size

    def transform_to_vehicle_coords(self, image_coeffs):
        """
        Transform a polynomial from image coords (x_img(y_img)) to vehicle coords (y_v(x_v)).

        Image Coordinate System:
        - Origin: (0,0) top-left
        - x-axis: to the right
        - y-axis: downward

        Vehicle Coordinate System:
        - Origin: corresponds to (x_img=256, y_img=287) in image coordinates
        - x_v-axis: upward relative to image (x_v = 287 - y_img)
        - y_v-axis: to the right relative to image (y_v = x_img - 256)

        Given:
        x_img(y_img) = a_n*y_img^n + a_(n-1)*y_img^(n-1) + ... + a_0
        Substitute y_img = 287 - x_v:
        x_img(x_v) = sum over i of a_(n-i)*(287 - x_v)^i

        Expand (287 - x_v)^i = sum_{k=0}^i [C(i,k)*287^(i-k)*(-1)^k * x_v^k]

        Collect terms by powers of x_v to get x_img in terms of x_v.
        Then shift by -256 to get y_v:
        y_v(x_v) = x_img(x_v) - 256

        Returns coefficients [b_n, b_(n-1), ..., b_0] of y_v(x_v).
        """
        if len(image_coeffs) < 3:
            image_coeffs = np.pad(image_coeffs, (3 - len(image_coeffs), 0), 'constant')

        n = len(image_coeffs) - 1
        vehicle_coeffs_asc = np.zeros(n+1)

        for j, a_j in enumerate(image_coeffs):
            p = n - j
            for k in range(p+1):
                term = a_j * comb(p, k) * (287**(p-k)) * ((-1)**k)
                vehicle_coeffs_asc[k] += term

        vehicle_coeffs_asc[0] -= 256
        vehicle_coeffs = vehicle_coeffs_asc[::-1]
        return vehicle_coeffs

    def calculate_errors(self, car_coeffs):
        """
        Calculate CTE and EPSI in vehicle coordinates.

        Given a polynomial in vehicle coords:
        y_v = b_n*x_v^n + ... + b_1*x_v + b_0

        At x_v=0:
        - y_v(0) = b_0
        - derivative dy_v/dx_v at 0 = b_1

        cte = b_0
        psides = arctan(b_1)
        epsi = -psides
        """
        b0 = car_coeffs[-1]
        b1 = car_coeffs[-2] if len(car_coeffs) > 1 else 0.0
        cte = b0
        psides = np.arctan(b1)
        epsi = -psides
        return cte, epsi

    def fit_polynomial(self, y_points, x_points, base_order):
        """
        Fits a polynomial to the provided points.
        Returns coefficients in descending order (like np.polyfit).
        """
        max_order = base_order + 2
        for order in range(base_order, max_order+1):
            try:
                coeffs = np.polyfit(y_points, x_points, order)
                return coeffs
            except np.RankWarning:
                continue
        return np.polyfit(y_points, x_points, max_order)

    def plan_path(self, lane_polynomials, vehicle_state):
        """
        1. Sample points between lanes -> x_mid
        2. Ensure (y=0,x=256) is included or polynomial passes through (y=287,x=256)
        3. Fit polynomial in image coords
        4. Transform to vehicle coords
        5. Compute cte, epsi from car_coeffs
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

        num_end_points = 4
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

        image_coeffs = self.fit_polynomial(y_vals, x_mid_points, self.order)
        car_coeffs = self.transform_to_vehicle_coords(image_coeffs)
        cte, epsi = self.calculate_errors(car_coeffs)

        return image_coeffs, cte, epsi, car_coeffs
