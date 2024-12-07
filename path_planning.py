import numpy as np

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
        Transform polynomial from image coords to vehicle coords.

        Vehicle coords:
        X' = x - w/2, Y' = y - 0 (Y'=y)

        If image polynomial: x = a0 + a1*y + a2*y² + ...
        X'(y) = (a0 + a1*y + ... ) - w/2

        Returns car_coeffs.
        """
        w = self.image_size[0]
        car_coeffs = image_coeffs.copy()
        car_coeffs[0] = car_coeffs[0] - w/2
        return car_coeffs

    def calculate_errors(self, car_coeffs):
        """
        Calculate CTE, EPSI in vehicle coords.
        X'(0) = car_coeffs[0]
        derivative at 0 = car_coeffs[1]
        epsidesired = arctan(car_coeffs[1])
        epsi = -epsidesired
        """
        A = car_coeffs
        cte = A[0]
        derivative_at_0 = A[1] if len(A) > 1 else 0.0
        theta_desired = np.arctan(derivative_at_0)
        epsi = -theta_desired
        return cte, epsi

    def fit_polynomial(self, y_points, x_points, base_order):
        """
        Try fitting a polynomial starting at base_order, and if not good, increase order.
        We'll limit the attempts to base_order+2 for safety.
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
        Steps:
        1. Sample points between lanes:
           - We know lane_polynomials are something like [a0,a1,a2] each (or more).
           - For y in a range, compute x_left, x_right from these polynomials.
           - midpoint x_mid = (x_left + x_right)/2
        2. Ensure the first point is (y=0, x=w/2).
        3. Fit a polynomial to (y, x_mid).
        4. Transform polynomial to vehicle coords for MPC (car_coeffs).
        5. Compute cte, epsi from car_coeffs.
        6. Return image_coeffs, cte, epsi, car_coeffs.
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

    def transform_to_vehicle_coords(self, image_coeffs):
        """
        Transform polynomial from image coords to vehicle coords.
        Vehicle at (w/2, 0)
        X'(y) = f(y) - w/2
        """
        w = self.image_size[0]
        car_coeffs = image_coeffs.copy()
        car_coeffs[0] = car_coeffs[0] - w/2
        return car_coeffs
