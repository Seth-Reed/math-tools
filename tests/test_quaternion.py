import unittest
import numpy as np
from dragonfly.functions.math.quaternion import normalize, inverse, multiply, rotate, quat2eul, eul2quat, angleaxis2quat, quat2angleaxis, error


class TestQuaternionOperations(unittest.TestCase):

    def test_normalize(self):
        q = np.array([2, 0, 0, 0])
        normalized_q = normalize(q)
        expected_result = np.array([1, 0, 0, 0])
        np.testing.assert_array_almost_equal(normalized_q, expected_result)

    def test_inverse(self):
        q = np.array([0, 1, 0, 0])
        inv_q = inverse(q)
        expected_result = np.array([0, -1, 0, 0])
        np.testing.assert_array_almost_equal(inv_q, expected_result)

    def test_multiply(self):
        q1 = np.array([1, 0, 0, 0])
        q2 = np.array([0, 1, 0, 0])
        result = multiply(q1, q2)
        expected_result = np.array([0, 1, 0, 0])
        np.testing.assert_array_almost_equal(result, expected_result)

    def test_rotate(self):
        vector = np.array([1, 0, 0])
        q = np.array([np.sqrt(2)/2, 0, np.sqrt(2)/2, 0])
        rotated_vector = rotate(vector, q, to_inertial=True)
        expected_result = np.array([0, 0, -1])
        np.testing.assert_array_almost_equal(rotated_vector, expected_result, decimal=5)

    def test_quat2eul_and_eul2quat(self):
        roll, pitch, yaw = np.pi/4, np.pi/6, np.pi/3
        q = eul2quat(roll, pitch, yaw)
        eul_angles = quat2eul(q)
        expected_eul_angles = np.array([yaw, pitch, roll])
        np.testing.assert_array_almost_equal(eul_angles, expected_eul_angles, decimal=5)

    def test_angleaxis2quat_and_quat2angleaxis(self):
        angle = np.pi/4
        axis = np.array([0, 1, 0])
        q = angleaxis2quat(angle, axis)
        recovered_angle, recovered_axis = quat2angleaxis(q)
        np.testing.assert_almost_equal(recovered_angle, angle)
        np.testing.assert_array_almost_equal(recovered_axis, axis)

    def test_error(self):
        q_i = np.array([1, 0, 0, 0])
        q_f = np.array([0, 1, 0, 0])
        err = error(q_i, q_f)
        expected_error = np.array([0, 1, 0, 0])
        np.testing.assert_array_almost_equal(err, expected_error)


if __name__ == '__main__':
    unittest.main()
