import unittest
import numpy as np
from path_planning import PathPlanningModule

class TestPathPlanningModule(unittest.TestCase):

    def setUp(self):
        self.module = PathPlanningModule(image_size=(512, 288))

    def test_transform_to_vehicle_coords_basic(self):
        """
        Sprawdza czy transform_to_vehicle_coords poprawnie
        przekształca wielomian przy użyciu default image_size=(512,288).
        """
        # Coeffs w image coords, np. a2=1, a1=2, a0=3
        image_coeffs = np.array([1.0, 2.0, 3.0])
        result = self.module.transform_to_vehicle_coords(image_coeffs)

        # Oczekiwane wzory:
        # b2 = a2
        # b1 = -(a1 + 2*(h - 1)*a2)
        # b0 = a0 + a1*(h - 1) + a2*(h - 1)^2 - (w / 2)
        # Dla h=288, w=512 => h-1=287, w/2=256
        expected_b2 = 1.0
        expected_b1 = -(2.0 + 2 * 287.0 * 1.0)  # = - (2 + 574) = -576
        expected_b0 = (3.0 + 2.0 * 287.0 + 1.0 * (287.0**2) - 256.0)
        # expected_b0 = 3 + 574 + 82369 - 256 = 3 + 574 + 82369 - 256
        # = 3 + 574 + 82113 = 82690

        self.assertAlmostEqual(result[0], expected_b2, places=5)
        self.assertAlmostEqual(result[1], expected_b1, places=5)
        self.assertAlmostEqual(result[2], expected_b0, places=5)

    def test_transform_to_vehicle_coords_with_padding(self):
        """
        Jeśli ktoś poda tylko 2 współczynniki, funkcja powinna je dopisać (pad)
        """
        image_coeffs = np.array([5.0, 10.0])
        result = self.module.transform_to_vehicle_coords(image_coeffs)

        w, h = (512, 288)
        h_minus_1 = h - 1
        half_w = w / 2

        # b2 = 0
        # b1 = -(5 + 2*(287)*0) = -5
        # b0 = 10 + 5*287 + 0*(287^2) - 256
        #    = 10 + 1435 - 256 = 1189
        self.assertAlmostEqual(result[0], 0.0, places=5)
        self.assertAlmostEqual(result[1], -5.0, places=5)
        self.assertAlmostEqual(result[2], 1189.0, places=5)

    def test_calculate_errors(self):
        """
        Test CTE i epsi dla prostego wielomianu w układzie pojazdu.
        """
        car_coeffs = np.array([0.0, 1.0, 10.0]) 
        cte, epsi = self.module.calculate_errors(car_coeffs)
        self.assertAlmostEqual(cte, 10.0, places=5)
        self.assertAlmostEqual(epsi, - (np.pi / 4), places=5)

    def test_plan_path_no_polynomials(self):
        """
        Plan path should return (None, None, None, None) if no polynomials are provided.
        """
        result = self.module.plan_path([], vehicle_state=[0,0,0,0,0,0])
        self.assertEqual(result, (None, None, None, None))

if __name__ == '__main__':
    unittest.main()
