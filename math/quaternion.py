import numpy as np


def normalize(q):
    """
    Normalizes a quaternion.

    Parameters:
        q (np.ndarray): Quaternion to be normalized.

    Returns:
        np.ndarray: Normalized quaternion.
    """
    norm_q = np.linalg.norm(q)
    if norm_q == 0.:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm_q


def inverse(q):
    """
    Computes the inverse of a unit quaternion.

    Parameters:
        q (np.ndarray): Quaternion represented as a 4-element array [q0, q1, q2, q3].

    Returns:
        np.ndarray: Inverse of the quaternion.
    """
    return normalize(np.array([q[0], -q[1], -q[2], -q[3]]))


def multiply(q, p):
    """
    Multiplies two unit quaternions. Order matters (e.g. p * q = multiply(q, p))

    Parameters:
        q (np.ndarray): First quaternion [q0, q1, q2, q3].
        p (np.ndarray): Second quaternion [p0, p1, p2, p3].

    Returns:
        np.ndarray: Product of the two unit quaternions
    """
    q0, q1, q2, q3 = q
    p0, p1, p2, p3 = p
    return np.array([p0 * q0 - p1 * q1 - p2 * q2 - p3 * q3,
                     p0 * q1 + p1 * q0 + p2 * q3 - p3 * q2,
                     p0 * q2 - p1 * q3 + p2 * q0 + p3 * q1,
                     p0 * q3 + p1 * q2 - p2 * q1 + p3 * q0])


def rotate(vector, quaternion, to_inertial=False):
    """
    Rotates a vector between body and inertial frames using a quaternion.

    Parameters:
        vector (np.ndarray): The 3D vector to rotate, specified as [x, y, z]. This can be
                             a body frame vector that needs to be expressed in the inertial frame
                             or vice versa.
        quaternion (np.ndarray): The unit quaternion representing the orientation of the body
                                 frame relative to the inertial frame, specified as [q0, q1, q2, q3].
        to_inertial (bool): If False (default), the vector is expressed in body frame and the
                            function returns its representation in the vehicle frame.
                            If True, the vector is expressed in vehicle frame and
                            the function returns its representation in the body frame.

    Returns:
        np.ndarray: The rotated vector. If to_inertial is True, returns the vector expressed in the inertial
                    frame; if False, returns the vector expressed in the body frame.
    """
    vector_quat = np.array([0] + list(vector))
    quaternion_inverse = inverse(quaternion)

    if to_inertial:
        # body to inertial: q_inv * vector_quat * q <-- compute right to left
        rotated_vector_quat = multiply(multiply(quaternion_inverse, vector_quat), quaternion)
    else:
        # inertial to body: q * vector_quat * q_inv <-- compute right to left
        rotated_vector_quat = multiply(multiply(quaternion, vector_quat), quaternion_inverse)

    return rotated_vector_quat[1:]


def quat2eul(quaternion):
    """
    Converts a quaternion into Euler angles (yaw, pitch, roll) using the ZYX rotation order.

    Parameters:
        quaternion (np.ndarray): Quaternion [q0, q1, q2, q3].

    Returns:
        np.ndarray: Euler angles [yaw, pitch, roll] in radians.
    """
    quaternion = normalize(quaternion)
    q0, q1, q2, q3 = quaternion

    yaw = np.arctan2(2.0 * (q0 * q3 + q1 * q2), 1.0 - 2.0 * (q2**2 + q3**2))
    pitch = np.arcsin(np.clip(2.0 * (q0 * q2 - q3 * q1), -1.0, 1.0))
    roll = np.arctan2(2.0 * (q0 * q1 + q2 * q3), 1.0 - 2.0 * (q1**2 + q2**2))

    return np.array([yaw, pitch, roll])


def eul2quat(roll=0., pitch=0., yaw=0., degrees=False):
    """
    Converts Euler angles (yaw, pitch, roll) into a quaternion using the ZYX rotation order.

    Parameters:
        roll (float): Roll angle. Defaults to 0. radians
        pitch (float): Pitch angle. Defaults to 0. radians
        yaw (float): Yaw angle. Defaults to 0. radians
        degrees (boolean): if False (default), the angles are assumed to be measured in radians. else, angles are in degrees.

    Returns:
        np.ndarray: Quaternion [q0, q1, q2, q3].
    """
    if degrees:
        roll  *= np.pi / 180.
        pitch *= np.pi / 180.
        yaw   *= np.pi / 180.

    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    q0 = cr * cp * cy + sr * sp * sy
    q1 = sr * cp * cy - cr * sp * sy
    q2 = cr * sp * cy + sr * cp * sy
    q3 = cr * cp * sy - sr * sp * cy

    return np.array([q0, q1, q2, q3])


def angleaxis2quat(angle, axis):
    """
    Converts an angle-axis representation to a quaternion.

    Parameters:
        angle (float): Rotation angle in radians.
        axis (np.ndarray): Rotation axis, must be a normalized 3D vector.

    Returns:
        np.ndarray: Quaternion [q0, q1, q2, q3].
    """
    axis = normalize(axis)
    q0 = np.cos(angle / 2.)
    q1 = axis[0] * np.sin(angle / 2.)
    q2 = axis[1] * np.sin(angle / 2.)
    q3 = axis[2] * np.sin(angle / 2.)

    return np.array([q0, q1, q2, q3])


def quat2angleaxis(quaternion):
    """
    Converts a quaternion to angle-axis representation.

    Parameters:
        quaternion (np.ndarray): Quaternion [q0, q1, q2, q3].

    Returns:
        tuple: (angle in radians, normalized axis [x, y, z]).
    """
    quaternion = normalize(quaternion)
    angle = 2 * np.arccos(quaternion[0])
    s = np.sqrt(1 - quaternion[0] ** 2)
    if s == 0.:
        # If s is close to zero, direction of axis is not important
        axis = np.array([1, 0, 0])
    else:
        axis = quaternion[1:] / s

    return angle, normalize(axis)


def error(initial_quaternion, final_quaternion):
    """
    Calculates the shortest relative rotation error between two quaternions.

    The error quaternion represents the rotation needed to align the first quaternion (q_i)
    with the second quaternion (q_f) and hence is expressed relative to the first quaternion.

    Parameters:
        initial_quaternion (np.ndarray): Initial quaternion [q0, q1, q2, q3].
        final_quaternion (np.ndarray): Final (target) quaternion [q0, q1, q2, q3].

    Returns:
        np.ndarray: Error quaternion representing the relative rotation from q_i to q_f.
    """

    # The error between two quaternions is computed as q_i^-1 * q_f,
    initial_quaternion_inverse = inverse(initial_quaternion)
    quaternion_error = multiply(final_quaternion, initial_quaternion_inverse)

    # return the quaternion error corresponding to the shortest rotation
    if quaternion_error[0] < 0:
        return inverse(quaternion_error)
    return quaternion_error







