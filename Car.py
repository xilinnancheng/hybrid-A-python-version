import numpy as np
from math import tan, pi, sin, cos, hypot
from scipy.spatial.transform import Rotation as R


class CarModel:
    def __init__(self):
        self.wheel_base = 2.84
        self.width_with_mirrors = 2.11
        self.front_bumper_to_origin = 3.89
        self.rear_bumper_to_origin = 1.043
        self.rear_axis_to_cg = 1.5
        self.max_steer = 0.5
        self.bubble_radius = hypot(max(
            self.front_bumper_to_origin - self.rear_axis_to_cg,
            self.rear_bumper_to_origin + self.rear_axis_to_cg), 0.5 * self.width_with_mirrors)

    def Simulate(self, x, y, yaw, steer, direction, distance):
        pos_sign = 1.0 if direction else -1.0
        updated_x = x + pos_sign * distance * cos(yaw)
        updated_y = y + pos_sign * distance * sin(yaw)
        updated_yaw = self.ToPi(
            yaw + pos_sign * distance * tan(steer)/self.wheel_base)
        return updated_x, updated_y, updated_yaw

    def ToPi(self, angle):
        return (angle + pi) % (2 * pi) - pi

    def CollisionCheck(self, x, y, yaw, ox, oy, environment_kd_tree):
        radius_x = x + self.rear_axis_to_cg * cos(yaw)
        radius_y = y + self.rear_axis_to_cg * sin(yaw)

        indexs = environment_kd_tree.query_ball_point(
            [radius_x, radius_y], self.bubble_radius)
        if not indexs:
            return False

        rot = R.from_euler('z', yaw).as_matrix()[0:2, 0:2]
        for index in indexs:
            obs_x_ref = ox[index] - x
            obs_y_ref = oy[index] - y

            rotated_obs_x_ref, rotated_obs_y_ref = np.matmul(
                rot, [obs_x_ref, obs_y_ref])

            if (rotated_obs_x_ref <= self.front_bumper_to_origin
                    and rotated_obs_x_ref >= -self.rear_bumper_to_origin
                    and rotated_obs_y_ref <= 0.5 * self.width_with_mirrors
                    and rotated_obs_y_ref >= -0.5 * self.width_with_mirrors):
                return True

        return False


if __name__ == "__main__":
    test_car = CarModel()
