import numpy as np
import math
import heapq
from Car import CarModel
import ReedsSheppPathPlanning as rs
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree


class Node:
    def __init__(self, x, y, yaw, x_index, y_index, yaw_index, direction, steer, parent_index=None, cost=None):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.x_index = x_index
        self.y_index = y_index
        self.yaw_index = yaw_index
        self.direction = direction
        self.steer = steer
        self.parent_index = parent_index
        self.cost = cost


class HybridAStarConfig:
    def __init__(self):
        self.xy_grid_resolution = 1.0
        self.yaw_grid_resolution = np.deg2rad(2.5)
        self.max_steer = 0.5
        self.n_steer = 10
        self.step = 1.5 * self.xy_grid_resolution
        self.switch_back_cost = 10.0
        self.back_motion_cost = 5.0
        self.steer_change_cost = 5.0
        self.steer_cost = 1.0


class HybridAStarPlanner:
    def __init__(self):
        self.config = HybridAStarConfig()
        self.obstacles_x = []
        self.obstacles_y = []
        self.environment_grid_min_xy = []
        self.environment_grid_max_xy = []
        self.environment_grid_min_yaw = 0
        self.environment_grid_max_yaw = 0
        self.environment_grid_width_x = 0
        self.environment_grid_width_y = 0
        self.environment_grid_width_yaw = 0
        self.environment_kd_tree = cKDTree(np.vstack(([1.0], [1.0])).T)

        self.car_model = CarModel()
        self.motion_input = self.CalMotionInput()
        self.path_x = []
        self.path_y = []

    def Environment(self):
        environment_upper_bound_x = 70.0
        environment_upper_bound_y = 15.0
        environment_lower_bound_x = 20.0
        environment_lower_bound_y = 0.0

        parking_slot_upper_bound_x = 50.0
        parking_slot_upper_bound_y = 8.0
        parking_slot_upper_lower_x = 40.0
        parking_slot_upper_lower_y = 4.0

        for i in np.arange(environment_upper_bound_y, parking_slot_upper_bound_y, -0.5):
            self.obstacles_x.append(environment_lower_bound_x)
            self.obstacles_y.append(i)
        for i in np.arange(environment_lower_bound_x, parking_slot_upper_lower_x, 0.5):
            self.obstacles_x.append(i)
            self.obstacles_y.append(parking_slot_upper_bound_y)
        for i in np.arange(parking_slot_upper_bound_y, parking_slot_upper_lower_y, -0.5):
            self.obstacles_x.append(parking_slot_upper_lower_x)
            self.obstacles_y.append(i)
        for i in np.arange(parking_slot_upper_lower_x, parking_slot_upper_bound_x, 0.5):
            self.obstacles_x.append(i)
            self.obstacles_y.append(parking_slot_upper_lower_y)
        for i in np.arange(parking_slot_upper_lower_y, parking_slot_upper_bound_y, 0.5):
            self.obstacles_x.append(parking_slot_upper_bound_x)
            self.obstacles_y.append(i)
        for i in np.arange(parking_slot_upper_bound_x, environment_upper_bound_x, 0.5):
            self.obstacles_x.append(i)
            self.obstacles_y.append(parking_slot_upper_bound_y)
        for i in np.arange(parking_slot_upper_bound_y, environment_upper_bound_y, 0.5):
            self.obstacles_x.append(environment_upper_bound_x)
            self.obstacles_y.append(i)
        for i in np.arange(environment_upper_bound_x, environment_lower_bound_x, -0.5):
            self.obstacles_x.append(i)
            self.obstacles_y.append(environment_upper_bound_y)

        self.environment_grid_min_xy = [
            round(min(self.obstacles_x)/self.config.xy_grid_resolution), round(min(self.obstacles_y)/self.config.xy_grid_resolution)]
        self.environment_grid_max_xy = [
            round(max(self.obstacles_x)/self.config.xy_grid_resolution), round(max(self.obstacles_y)/self.config.xy_grid_resolution)]
        self.environment_grid_min_yaw = round(
            - math.pi / self.config.yaw_grid_resolution) - 1
        self.environment_grid_max_yaw = round(
            math.pi / self.config.yaw_grid_resolution)

        self.environment_grid_width_x = (self.environment_grid_max_xy[0] -
                                         self.environment_grid_min_xy[0])
        self.environment_grid_width_y = (self.environment_grid_max_xy[1] -
                                         self.environment_grid_min_xy[1])
        self.environment_grid_width_yaw = self.environment_grid_max_yaw - \
            self.environment_grid_min_yaw
        self.environment_kd_tree = cKDTree(
            np.vstack((self.obstacles_x, self.obstacles_y)).T)

    def CalcNodeIndex(self, node):
        index = (node.yaw_index - self.environment_grid_min_yaw) * self.environment_grid_width_x * self.environment_grid_width_y + \
            (node.x_index - self.environment_grid_min_xy[0]
             ) * self.environment_grid_width_y + (node.y_index - self.environment_grid_min_xy[1])
        return index

    def CalMotionInput(self):
        motion_input = []
        for steer in np.concatenate(((np.linspace(-self.config.max_steer, self.config.max_steer, self.config.n_steer)), np.array([0.0]))):
            for direction in [-1, 1]:
                motion_input.append((steer, direction))
        return motion_input

    def GetNeighbourNodes(self, node, node_index):
        neighbour_nodes = []
        for steer, direction in self.motion_input:
            neighbour_x, neighbour_y, neighbour_yaw = self.car_model.Simulate(
                node.x, node.y, node.yaw, steer, direction, self.config.step)
            if self.car_model.CollisionCheck(neighbour_x, neighbour_y, neighbour_yaw,
                                             self.obstacles_x, self.obstacles_y, self.environment_kd_tree):
                continue

            neighbour_x_index = round(
                neighbour_x/self.config.xy_grid_resolution)
            neighbour_y_index = round(
                neighbour_y/self.config.xy_grid_resolution)

            if (neighbour_x_index <= self.environment_grid_min_xy[0]
                or neighbour_x_index >= self.environment_grid_max_xy[0]
                or neighbour_y_index <= self.environment_grid_min_xy[1]
                    or neighbour_y_index >= self.environment_grid_max_xy[1]):
                continue

            neighbour_yaw_index = round(
                neighbour_yaw/self.config.yaw_grid_resolution)

            neighbour_node = Node(
                neighbour_x, neighbour_y, neighbour_yaw,
                neighbour_x_index, neighbour_y_index,
                neighbour_yaw_index, direction, steer, node_index, node.cost + self.config.step)

            neighbour_node.cost += self.config.switch_back_cost if node.direction != neighbour_node.direction else 0.0
            neighbour_node.cost += self.config.steer_cost * steer
            neighbour_node.cost += self.config.steer_change_cost * \
                abs(steer - node.steer)

            neighbour_nodes.append(neighbour_node)
        return neighbour_nodes

    def GetEuclideanH(self, curr_node, goal_node):
        return math.hypot(curr_node.x - goal_node.x, curr_node.y - goal_node.y)

    def GetReedSheppCost(self, path):
        path_cost = 0.0
        for length in path.lengths:
            if length >= 0.0:
                path_cost += length
            else:
                path_cost += abs(length) * self.config.back_motion_cost

        for i in range(len(path.lengths) - 1):
            if path.lengths[i] * path.lengths[i+1] < 0.0:
                path_cost += self.config.switch_back_cost

        for course_type in path.ctypes:
            if course_type != "S":  # curve
                path_cost += self.config.steer_cost * \
                    abs(self.config.max_steer)

        n_ctypes = len(path.ctypes)
        u_list = [0.0] * n_ctypes
        for i in range(n_ctypes):
            if path.ctypes[i] == "R":
                u_list[i] = - self.config.max_steer
            elif path.ctypes[i] == "L":
                u_list[i] = self.config.max_steer

        for i in range(len(path.ctypes) - 1):
            path_cost += self.config.steer_change_cost * \
                abs(u_list[i + 1] - u_list[i])

        return path_cost

    def TryReedSheppPath(self, curr_node, goal_node):
        max_curvature = math.tan(self.config.max_steer) / \
            self.car_model.wheel_base
        paths = rs.calc_paths(curr_node.x, curr_node.y, curr_node.yaw, goal_node.x,
                              goal_node.y, goal_node.yaw, max_curvature, step_size=self.config.step)
        if not paths:
            return False, None

        best_cost, best_path = None, None
        for path in paths:
            is_collision = False
            for i in range(len(path.x)):
                x = path.x[i]
                y = path.y[i]
                yaw = path.yaw[i]
                if(self.car_model.CollisionCheck(x, y, yaw,
                                                 self.obstacles_x, self.obstacles_y, self.environment_kd_tree)):
                    is_collision = True
                    break

            if not is_collision:
                cost = self.GetReedSheppCost(path)
                if not best_cost or best_cost > cost:
                    best_path = path
                    best_cost = cost
        return False if not best_cost else True, best_path

    def GetPath(self, closed_list, goal_node, find_rs_path, path):
        curr_node = goal_node
        while curr_node.parent_index != -1:
            self.path_x.append(curr_node.x)
            self.path_y.append(curr_node.y)
            curr_node = closed_list[curr_node.parent_index]
        self.path_x.reverse()
        self.path_y.reverse()

        if find_rs_path:
            self.path_x.extend(path.x)
            self.path_y.extend(path.y)

    def Plan(self, start, goal, show_animation=False, use_rs_curve=False):
        self.Environment()

        start_node = Node(start[0], start[1], start[2], round(start[0]/self.config.xy_grid_resolution), round(
            start[1]/self.config.xy_grid_resolution), round(start[2]/self.config.yaw_grid_resolution), True, 0.0, -1, 0.0)
        goal_node = Node(goal[0], goal[1], goal[2], round(goal[0]/self.config.xy_grid_resolution), round(
            goal[1]/self.config.xy_grid_resolution), round(goal[2]/self.config.yaw_grid_resolution), True, 0.0, -1)

        h_pq = []
        open_list = {}
        closed_list = {}

        open_list[self.CalcNodeIndex(start_node)] = start_node
        heapq.heappush(h_pq, (start_node.cost, self.CalcNodeIndex(start_node)))

        while True:
            if not open_list:
                print("Warning: Could not find a path")
                return

            curr_node_cost, curr_node_index = heapq.heappop(h_pq)
            if curr_node_index in open_list:
                curr_node = open_list[curr_node_index]
            else:
                continue

            open_list.pop(curr_node_index)
            closed_list[curr_node_index] = curr_node

            if show_animation:
                if curr_node.direction == 1:
                    plt.arrow(x=curr_node.x, y=curr_node.y, dx=1.0 * math.cos(
                        curr_node.yaw), dy=1.0 * math.sin(curr_node.yaw), width=.08, color='k')
                else:
                    plt.arrow(x=curr_node.x, y=curr_node.y, dx=1.0 * math.cos(
                        curr_node.yaw), dy=1.0 * math.sin(curr_node.yaw), width=.08, color='b')

            find_rs_path, path = None, None
            if use_rs_curve:
                find_rs_path, path = self.TryReedSheppPath(
                    curr_node, goal_node)

            if (curr_node.x_index == goal_node.x_index and curr_node.y_index == goal_node.y_index
                    and abs(curr_node.yaw_index - goal_node.yaw_index) <= 1) or find_rs_path:
                final_goal_node = curr_node
                print("Info: Find a path")
                break

            for neighbour_node in self.GetNeighbourNodes(curr_node, curr_node_index):
                neighbour_node_index = self.CalcNodeIndex(neighbour_node)
                if neighbour_node_index in closed_list:
                    continue

                if neighbour_node_index not in open_list \
                        or neighbour_node.cost <= open_list[neighbour_node_index].cost:
                    open_list[neighbour_node_index] = neighbour_node
                    heapq.heappush(
                        h_pq, (neighbour_node.cost + self.GetEuclideanH(neighbour_node, goal_node), neighbour_node_index))

        self.GetPath(closed_list, final_goal_node, find_rs_path, path)


def main():
    print("Hybrid A* Planning Demo")
    start_pose = [30, 10, np.deg2rad(0.0)]
    goal_pose = [45, 7, np.deg2rad(0.0)]

    hybrid_a_star_planner = HybridAStarPlanner()
    hybrid_a_star_planner.Plan(
        start_pose, goal_pose, show_animation=True, use_rs_curve=False)

    plt.xlabel("x coordinate")
    plt.ylabel("y coordinate")
    plt.plot(hybrid_a_star_planner.obstacles_x,
             hybrid_a_star_planner.obstacles_y, 'k')
    plt.plot(hybrid_a_star_planner.path_x,
             hybrid_a_star_planner.path_y, 'b')
    plt.arrow(x=start_pose[0], y=start_pose[1], dx=2.0 * math.cos(
        start_pose[2]), dy=2.0 * math.sin(start_pose[2]), width=.08, color='b')
    plt.arrow(x=goal_pose[0], y=goal_pose[1], dx=2.0 * math.cos(
        goal_pose[2]), dy=2.0 * math.sin(goal_pose[2]), width=.08, color='r')
    plt.title("Side Parking Scenario")
    plt.savefig('side parking_without_rs_curve_resolution_1.0m.png')
    plt.show()


if __name__ == "__main__":
    main()
