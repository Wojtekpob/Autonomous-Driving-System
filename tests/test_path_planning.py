import unittest
import numpy as np
from path_planning import PathPlanningModule

class TestPathPlanningModule(unittest.TestCase):

    def setUp(self):
        self.module = PathPlanningModule(image_size=(512, 288))

    def test_transform_to_vehicle_coords_basic(self):
        image_coeffs = np.array([1.0, 2.0, 3.0])
        result = self.module.transform_to_vehicle_coords(image_coeffs)

        expected_b2 = 1.0
        expected_b1 = -(2.0 + 2 * 287.0 * 1.0) 
        expected_b0 = (3.0 + 2.0 * 287.0 + 1.0 * (287.0**2) - 256.0)

        self.assertAlmostEqual(result[0], expected_b2, places=5)
        self.assertAlmostEqual(result[1], expected_b1, places=5)
        self.assertAlmostEqual(result[2], expected_b0, places=5)

    def test_transform_to_vehicle_coords_with_padding(self):
        image_coeffs = np.array([5.0, 10.0])
        result = self.module.transform_to_vehicle_coords(image_coeffs)

        w, h = (512, 288)
        h_minus_1 = h - 1
        half_w = w / 2

        self.assertAlmostEqual(result[0], 0.0, places=5)
        self.assertAlmostEqual(result[1], -5.0, places=5)
        self.assertAlmostEqual(result[2], 1189.0, places=5)

    def test_calculate_errors(self):
        car_coeffs = np.array([0.0, 1.0, 10.0]) 
        cte, epsi = self.module.calculate_errors(car_coeffs)
        self.assertAlmostEqual(cte, 10.0, places=5)
        self.assertAlmostEqual(epsi, - (np.pi / 4), places=5)

    def test_plan_path_no_polynomials(self):
        result = self.module.plan_path([], vehicle_state=[0,0,0,0,0,0])
        self.assertEqual(result, (None, None, None, None))

if __name__ == '__main__':
    unittest.main()
