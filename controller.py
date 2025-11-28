import numpy as np
from numpy.typing import ArrayLike

from simulator import RaceTrack
import math


class PIDController:
    def __init__(self, **kwargs):
        self._kp: float = kwargs.get("kp", 0)
        self._ki: float = kwargs.get("ki", 0)
        self._kd: float = kwargs.get("kd", 0)
        # csv to write to
        self._errors: list[float] = []
        self._outputs: list[float] = []

        self._integral: float = 0
        self._prev_error: float = 0

    def update(self, error: float):
        self._integral += error
        self._integral = np.clip(self._integral, -4.0, 4.0)

        derivative = error - self._prev_error
        self._prev_error = error
        output = self._kp * error + self._ki * self._integral + self._kd * derivative

        return output


steer_pid = PIDController(kp=10)
accel_pid = PIDController(kp=10)


def lower_controller(state, desired, parameters):
    s_x, s_y, delta, v, phi = state

    desired_steering_angle, desired_velocity = desired
    v_delta = steer_pid.update(desired_steering_angle - delta)
    accel = accel_pid.update(desired_velocity - v)

    # [steering angle change, acceleration]
    return np.array([v_delta, accel]).T


def get_curvature(p0, p1, p2, norm_factor=0.3):
    v1 = p1 - p0
    v2 = p2 - p1

    # angle between segments
    angle = math.atan2(np.cross(v1, v2), np.dot(v1, v2))
    curvature = abs(angle)

    # curvature factor:
    # 0.0 = straight, >0.3 = tight turn
    curv_norm = np.clip(curvature / norm_factor, 0, 1)
    return curv_norm


def controller(
    state: ArrayLike, parameters: ArrayLike, racetrack: RaceTrack
) -> ArrayLike:

    s_x, s_y, delta, v, phi = state

    pos = np.array([s_x, s_y])

    # distance from car to every point
    d = np.linalg.norm(racetrack.centerline - pos, axis=1)
    closest = np.argmin(d)

    # measure upcoming curvature
    p0 = racetrack.centerline[closest]
    p1 = racetrack.centerline[(closest + 10) % len(racetrack.centerline)]
    p2 = racetrack.centerline[(closest + 20) % len(racetrack.centerline)]

    curv_norm = get_curvature(p0, p1, p2)
    lookahead = 15.0 + (1 - curv_norm) * 5.0

    before = (closest - 1) % len(racetrack.centerline)
    after = (closest + 1) % len(racetrack.centerline)

    # decide whether the car is "heading" toward after or before
    if d[after] < d[before]:
        cur = closest
    else:
        cur = before

    remaining = lookahead

    while True:
        nxt = (cur + 1) % len(racetrack.centerline)

        # distance between consecutive centerline points
        seg = np.linalg.norm(racetrack.centerline[nxt] - racetrack.centerline[cur])

        if seg >= remaining:
            # interpolate between cur and nxt
            ratio = remaining / seg
            p = (1 - ratio) * racetrack.centerline[cur] + ratio * racetrack.centerline[
                nxt
            ]
            p_x, p_y = p[0], p[1]
            break

        # otherwise continue to next segment
        remaining -= seg
        cur = nxt

        # p_x, p_y now contain the final lookahead point

    racetrack.desired = p_x, p_y

    # find heading to go from (s_x, s_y) to (p_x, p_y)

    phi_desired = np.atan2(p_y - s_y, p_x - s_x)
    dx = p_x - s_x
    dy = p_y - s_y

    alpha = math.atan2(dy, dx) - phi
    alpha = (alpha + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]

    wheelbase = parameters[0]
    steering_angle = math.atan2(2 * wheelbase * math.sin(alpha), lookahead)

    steering_angle = np.clip(steering_angle, -0.5, 0.5)

    # base speed
    v_min = 15
    v_max = 50

    velocity = v_max - curv_norm**2 * (v_max - v_min)

    q0 = racetrack.centerline[(closest + 30) % len(racetrack.centerline)]
    q1 = racetrack.centerline[(closest + 40) % len(racetrack.centerline)]
    q2 = racetrack.centerline[(closest + 50) % len(racetrack.centerline)]

    upcoming_curv_norm = get_curvature(q0, q1, q2)

    if abs(steering_angle) < 0.05 and upcoming_curv_norm < 0.1:
        velocity *= 1.5

    # [steering angle, velocity]
    return np.array([steering_angle, velocity]).T
