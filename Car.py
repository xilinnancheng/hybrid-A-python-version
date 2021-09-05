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
        self.relative_contour = [np.array([self.front_bumper_to_origin,
                                           0.5 * self.width_with_mirrors]),
                                 np.array([-self.rear_bumper_to_origin,
                                           0.5 * self.width_with_mirrors]),
                                 np.array([-self.rear_bumper_to_origin,
                                           -0.5 * self.width_with_mirrors]),
                                 np.array([self.front_bumper_to_origin,
                                           -0.5 * self.width_with_mirrors])]

    def Simulate(self, x, y, yaw, steer, direction, distance):
        pos_sign = 1.0 if direction == 1 else -1.0
        updated_x = x + pos_sign * distance * cos(yaw)
        updated_y = y + pos_sign * distance * sin(yaw)
        updated_yaw = self.ToPi(
            yaw + pos_sign * distance * tan(steer)/self.wheel_base)
        return updated_x, updated_y, updated_yaw

    def ToPi(self, angle):
        return (angle + pi) % (2 * pi) - pi

    def CircleCollisionCheck(self, x, y, yaw, environment_kd_tree):
        radius_x = x + self.rear_axis_to_cg * cos(yaw)
        radius_y = y + self.rear_axis_to_cg * sin(yaw)

        indexs = environment_kd_tree.query_ball_point(
            [radius_x, radius_y], self.bubble_radius)
        return indexs

    # Collision Check Method 1
    def CollisionCheck(self, x, y, yaw, ox, oy, environment_kd_tree):
        indexs = self.CircleCollisionCheck(x, y, yaw, environment_kd_tree)
        if not indexs:
            return False

        rot = R.from_euler('z', yaw).as_matrix()[0:2, 0:2]
        for index in indexs:
            obs_x_ref = ox[index] - x
            obs_y_ref = oy[index] - y

            rotated_obs_x_ref, rotated_obs_y_ref = np.matmul(
                rot, [obs_x_ref, obs_y_ref])

            if (rotated_obs_x_ref <= self.front_bumper_to_origin + 1e-10
                    and rotated_obs_x_ref >= -self.rear_bumper_to_origin - 1e-10
                    and rotated_obs_y_ref <= 0.5 * self.width_with_mirrors + 1e-10
                    and rotated_obs_y_ref >= -0.5 * self.width_with_mirrors - 1e-10):
                return True

        return False

    # Collision Check Method 2
    def ConvexCollisionCheck(self, x, y, yaw, ox, oy, environment_kd_tree):
        indexs = self.CircleCollisionCheck(x, y, yaw, environment_kd_tree)
        if not indexs:
            return False

        rot = R.from_euler('z', yaw).as_matrix()[0:2, 0:2]
        for index in indexs:
            obs_x_ref = ox[index] - x
            obs_y_ref = oy[index] - y

            rotated_obs_x_ref, rotated_obs_y_ref = np.matmul(
                rot, [obs_x_ref, obs_y_ref])
            is_collision = None
            for i in range(len(self.relative_contour)):
                next_i = i+1 if i != 3 else 0
                vec_1 = np.array(
                    [rotated_obs_x_ref, rotated_obs_y_ref]) - self.relative_contour[i]
                vec_2 = self.relative_contour[next_i] - \
                    self.relative_contour[i]

                cross_result = np.cross(vec_1, vec_2)
                is_collision = (
                    cross_result > -1e-7) if is_collision == None else (is_collision ^ (cross_result > -1e-7))
                if is_collision == 1:
                    break

            if is_collision == 0:
                return True

        return False

    # Collision Check Method 3
    def ScanLineCollisionCheck(self, x, y, yaw, ox, oy, environment_kd_tree):
        indexs = self.CircleCollisionCheck(x, y, yaw, environment_kd_tree)
        if not indexs:
            return False

        rot = R.from_euler('z', yaw).as_matrix()[0:2, 0:2]
        for index in indexs:
            obs_x_ref = ox[index] - x
            obs_y_ref = oy[index] - y

            rotated_obs_x_ref, rotated_obs_y_ref = np.matmul(
                rot, [obs_x_ref, obs_y_ref])
            is_collision = False
            for i in range(len(self.relative_contour)):
                next_i = i+1 if i != 3 else 0

                vec_1 = np.array(
                    [rotated_obs_x_ref, rotated_obs_y_ref]) - self.relative_contour[i]
                vec_2 = np.array(
                    [rotated_obs_x_ref, rotated_obs_y_ref]) - self.relative_contour[next_i]

                if abs(np.cross(vec_1, vec_2)) < 1e-7 and np.dot(vec_1, vec_2) <= 0:
                    return True

                if rotated_obs_y_ref <= max(self.relative_contour[i][1], self.relative_contour[next_i][1]) - 1e-7\
                        and rotated_obs_y_ref > min(self.relative_contour[i][1], self.relative_contour[next_i][1]) + 1e-7:
                    if abs(self.relative_contour[next_i][1] - self.relative_contour[i][1]) < 1e-7:
                        if self.relative_contour[next_i][0] > rotated_obs_x_ref:
                            is_collision = not is_collision
                    elif ((self.relative_contour[i][0] + (rotated_obs_y_ref - self.relative_contour[i][1]) * (self.relative_contour[next_i][0] - self.relative_contour[i][0])/(self.relative_contour[next_i][1] - self.relative_contour[i][1])) > rotated_obs_x_ref):
                        is_collision = not is_collision

            if is_collision:
                return True
        return False

    # Collision Check Method 4
    def RotateAngelCollisionCheck(self, x, y, yaw, ox, oy, environment_kd_tree):
        indexs = self.CircleCollisionCheck(x, y, yaw, environment_kd_tree)
        if not indexs:
            return False

        rot = R.from_euler('z', yaw).as_matrix()[0:2, 0:2]
        for index in indexs:
            obs_x_ref = ox[index] - x
            obs_y_ref = oy[index] - y

            rotated_obs_x_ref, rotated_obs_y_ref = np.matmul(
                rot, [obs_x_ref, obs_y_ref])
            wn = 0
            for i in range(len(self.relative_contour)):
                next_i = i+1 if i != 3 else 0

                vec_1 = np.array(
                    [rotated_obs_x_ref, rotated_obs_y_ref]) - self.relative_contour[i]
                vec_2 = np.array(
                    [rotated_obs_x_ref, rotated_obs_y_ref]) - self.relative_contour[next_i]

                if abs(np.cross(vec_1, vec_2)) < 1e-7 and np.dot(vec_1, vec_2) <= 0:
                    return True

                vec_3 = self.relative_contour[next_i] - \
                    self.relative_contour[i]
                is_left = np.cross(vec_3, vec_1) > 0.0
                dist_1 = rotated_obs_y_ref - self.relative_contour[i][1]
                dist_2 = rotated_obs_y_ref - self.relative_contour[next_i][1]
                if is_left and dist_2 > 0 and dist_1 <= 0:
                    wn -= 1
                if not is_left and dist_2 <= 0 and dist_1 > 0:
                    wn += 1

            if wn != 0:
                return True
        return False


if __name__ == "__main__":
    test_car = CarModel()
