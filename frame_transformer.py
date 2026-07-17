"""
Transform force/torque readings from the TCP frame to the robot Base frame.

Uses the current TCP orientation (roll, pitch, yaw) to build a rotation
matrix and rotate the 6-DOF sensor vector into the Base frame so that
velocity commands via speed_x are expressed correctly.
"""

import math
from typing import Tuple


# 6-DOF force/torque tuple
FT6 = Tuple[float, float, float, float, float, float]


def rotation_matrix_zyx(roll: float, pitch: float, yaw: float):
    """Build a 3x3 rotation matrix from ZYX Euler angles (in radians).

    Convention:  R = Rz(yaw) * Ry(pitch) * Rx(roll)

    When (roll, pitch, yaw) describe the TCP orientation in the Base frame,
    this matrix rotates a vector **from TCP frame to Base frame**.

    Returns a 3x3 nested list  ``R[row][col]``.
    """
    cr, sr = math.cos(roll),  math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw),   math.sin(yaw)

    return [
        [cy * cp,   cy * sp * sr - sy * cr,   cy * sp * cr + sy * sr],
        [sy * cp,   sy * sp * sr + cy * cr,   sy * sp * cr - cy * sr],
        [-sp,       cp * sr,                  cp * cr               ],
    ]


def _mat_vec(R, v):
    """Multiply 3x3 matrix *R* by 3-vector *v*."""
    return (
        R[0][0] * v[0] + R[0][1] * v[1] + R[0][2] * v[2],
        R[1][0] * v[0] + R[1][1] * v[1] + R[1][2] * v[2],
        R[2][0] * v[0] + R[2][1] * v[1] + R[2][2] * v[2],
    )


class FrameTransformer:
    """Rotate a 6-DOF force/torque vector from TCP frame to Base frame.

    Typical usage inside a control loop::

        ft = FrameTransformer()
        tcp_pose = r.get_tcp_pose()          # [x,y,z,roll,pitch,yaw] (rad)
        ft.set_orientation(tcp_pose[3], tcp_pose[4], tcp_pose[5])
        fx_b, fy_b, fz_b, mx_b, my_b, mz_b = ft.transform(fx, fy, fz, mx, my, mz)

    Parameters
    ----------
    roll, pitch, yaw : float
        Initial TCP orientation in **radians**.  Updated later via
        :meth:`set_orientation`.
    """

    def __init__(self, roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0) -> None:
        self._R = rotation_matrix_zyx(roll, pitch, yaw)

    def set_orientation(self, roll: float, pitch: float, yaw: float) -> None:
        """Recompute the rotation matrix for a new TCP orientation."""
        self._R = rotation_matrix_zyx(roll, pitch, yaw)

    def rotation(self):
        """Current 3x3 TCP→Base rotation as nested lists R[row][col]."""
        return self._R

    def transform(
        self,
        fx: float, fy: float, fz: float,
        mx: float, my: float, mz: float,
    ) -> FT6:
        """Rotate forces and torques from TCP frame into Base frame.

        Returns
        -------
        (fx_base, fy_base, fz_base, mx_base, my_base, mz_base)
        """
        f_base = _mat_vec(self._R, (fx, fy, fz))
        m_base = _mat_vec(self._R, (mx, my, mz))
        return (*f_base, *m_base)
