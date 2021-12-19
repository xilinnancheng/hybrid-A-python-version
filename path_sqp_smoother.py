import osqp
import math
import numpy as np
from scipy import sparse
from matplotlib import pyplot as plt


# f(x) = 1/2 * x^T * Q * x + p^T * x
class SQPSmoother:
    def __init__(self, points_x, points_y, lb, ub, curvature_limit, smooth_weight, length_weight, distance_weight, slack_weight, fixed_start_end):
        self.num_of_points_ = len(points_x)
        self.num_of_slack_variables_ = self.num_of_points_ - 2
        self.num_of_pos_variables = 2 * self.num_of_points_
        self.num_of_variables_ = self.num_of_pos_variables + self.num_of_slack_variables_
        self.smooth_weight_ = smooth_weight
        self.length_weight_ = length_weight
        self.distance_weight_ = distance_weight
        self.slack_weight_ = slack_weight
        self.x_ref_ = points_x
        self.y_ref_ = points_y
        self.curvature_limit_ = curvature_limit
        self.lb_ = lb
        self.ub_ = ub
        self.fixed_start_end_ = fixed_start_end

        self.sqp_pen_max_iter_ = 5
        self.sqp_ftol_ = 1e-2
        self.sqp_sub_max_iter_ = 5
        self.sqp_ctol_ = 1

        total_length = 0.0
        for i in range(self.num_of_points_ - 1):
            total_length += math.hypot(self.x_ref_[i+1] - self.x_ref_[i],
                                       self.y_ref_[i+1] - self.y_ref_[i])
        self.average_length = total_length / (self.num_of_points_ - 1)
        # print(self.average_length)

    def CalculateSubQ(self, index_offset):
        row = []
        col = []
        data = []

        # first two rows
        for i in range(0, 3):
            row.append(index_offset)
            col.append(index_offset + i)
            if i == 0:
                data.append(self.smooth_weight_ +
                            self.length_weight_ + self.distance_weight_)
            elif i == 1:
                data.append(-2.0 * self.smooth_weight_ -
                            self.length_weight_)
            elif i == 2:
                data.append(self.smooth_weight_)

        if self.num_of_points_ > 3:
            for i in range(0, 3):
                row.append(index_offset + 1)
                col.append(index_offset + 1 + i)
                if i == 0:
                    data.append(5.0 * self.smooth_weight_ +
                                2.0 * self.length_weight_ + self.distance_weight_)
                elif i == 1:
                    data.append(-4.0 * self.smooth_weight_ -
                                self.length_weight_)
                elif i == 2:
                    data.append(self.smooth_weight_)

        # middle rows
        for i in range(index_offset + 2, index_offset + self.num_of_points_ - 2):
            for j in range(0, 3):
                row.append(i)
                col.append(i + j)
                if j == 0:
                    data.append(6*self.smooth_weight_ + 2 *
                                self.length_weight_ + self.distance_weight_)
                elif j == 1:
                    data.append(-4.0 * self.smooth_weight_ -
                                self.length_weight_)
                elif j == 2:
                    data.append(self.smooth_weight_)

        # last two rows
        for i in range(0, 2):
            row.append(index_offset + self.num_of_points_ - 2)
            col.append(index_offset + self.num_of_points_ - 2 + i)
            if i == 0:
                data.append(5.0 * self.smooth_weight_ +
                            2.0 * self.length_weight_ + self.distance_weight_)
            elif i == 1:
                data.append(-2.0 * self.smooth_weight_ -
                            self.length_weight_)

        row.append(index_offset + self.num_of_points_ - 1)
        col.append(index_offset + self.num_of_points_ - 1)
        data.append(self.smooth_weight_ +
                    self.length_weight_ + self.distance_weight_)

        return row, col, data

    def CalculateQ(self):
        x_row, x_col, x_data = self.CalculateSubQ(0)
        y_row, y_col, y_data = self.CalculateSubQ(self.num_of_points_)

        x_row.extend(y_row)
        x_col.extend(y_col)
        x_data.extend(y_data)

        row = np.array(x_row)
        col = np.array(x_col)
        data = np.array(x_data)

        for i in range(len(row)):
            data[i] *= 2.0

        Q = sparse.csc_matrix((data, (row, col)), shape=(
            self.num_of_variables_, self.num_of_variables_))
        # print(Q.toarray())
        return Q

    def CalculateP(self):
        P = []
        for i in range(self.num_of_points_):
            P.append(-2.0 * self.distance_weight_ * self.x_ref_[i])

        for i in range(self.num_of_points_):
            P.append(-2.0 * self.distance_weight_ * self.y_ref_[i])

        for i in range(self.num_of_slack_variables_):
            P.append(self.slack_weight_)

        P = np.array(P)
        # print(P)
        return P

    def CalculateSmooth(self, xi_minus_1, xi, xi_plus_1, yi_minus_1, yi, yi_plus_1):
        return math.pow(xi_minus_1 + xi_plus_1 - 2 * xi, 2.0) + \
            math.pow(yi_minus_1 + yi_plus_1 - 2 * yi, 2.0)

    def CalculateConstraintViolation(self, x, y):
        total_length = 0.0
        for i in range(len(x) - 1):
            total_length += math.hypot(x[i + 1] - x[i], y[i + 1] - y[i])

        average_interval_length = total_length / (len(x) - 1)
        curvature_constraint_sqr = pow(
            pow(average_interval_length, 2) * self.curvature_limit_, 2)
        max_violation = -1e20

        for i in range(1, len(x) - 1):
            x_pre = x[i - 1]
            x_cur = x[i]
            x_nex = x[i + 1]

            y_pre = y[i - 1]
            y_cur = y[i]
            y_nex = y[i + 1]
            violation = abs(curvature_constraint_sqr -
                            pow(x_pre + x_nex - 2.0 * x_cur, 2.0) -
                            pow(y_pre + y_nex - 2.0 * y_cur, 2.0))
            max_violation = violation if violation > max_violation else max_violation
        return max_violation

    def CalculateAffineConstraint(self, x, y):
        row = []
        col = []
        data = []

        lb = []
        ub = []

        for i in range(self.num_of_variables_):
            row.append(i)
            col.append(i)
            data.append(1.0)

            if i < self.num_of_points_:
                lb.append(x[i] - self.lb_)
                ub.append(x[i] + self.ub_)
            elif i < self.num_of_pos_variables:
                lb.append(
                    y[i - self.num_of_pos_variables] - self.lb_)
                ub.append(
                    y[i - self.num_of_pos_variables] + self.ub_)
            else:
                lb.append(0.0)
                ub.append(1e10)

        row_num = self.num_of_variables_
        for i in range(1, self.num_of_points_ - 1):
            common_x_val = x[i-1] +\
                x[i+1] - 2*x[i]
            row.append(row_num)
            col.append(i - 1)
            delta_f1 = 2.0 * common_x_val
            data.append(delta_f1)

            row.append(row_num)
            col.append(i)
            delta_f2 = -4.0 * common_x_val
            data.append(delta_f2)

            row.append(row_num)
            col.append(i+1)
            delta_f3 = 2.0 * common_x_val
            data.append(delta_f3)

            common_y_val = y[i-1] + \
                y[i+1] - 2*y[i]
            row.append(row_num)
            col.append(self.num_of_points_ + i - 1)
            delta_f4 = 2.0 * common_y_val
            data.append(delta_f4)

            row.append(row_num)
            col.append(self.num_of_points_ + i)
            delta_f5 = -4.0 * common_y_val
            data.append(delta_f5)

            row.append(row_num)
            col.append(self.num_of_points_ + i + 1)
            delta_f6 = 2.0 * common_y_val
            data.append(delta_f6)

            delta_f = np.array(
                [delta_f1, delta_f2, delta_f3, delta_f4, delta_f5, delta_f6])
            xy_ref = np.array([x[i-1], x[i], x[i+1],
                               y[i-1], y[i], y[i+1]])

            ub.append(math.pow(math.pow(self.average_length, 2.0)
                      * self.curvature_limit_, 2.0) -
                      self.CalculateSmooth(x[i-1], x[i], x[i+1],
                                           y[i-1], y[i], y[i+1]) +
                      np.dot(delta_f, xy_ref))
            lb.append(-1e10)
            row_num += 1

        A = sparse.csc_matrix((data, (row, col)), shape=(
            self.num_of_variables_ + self.num_of_slack_variables_, self.num_of_variables_))
        lb = np.array(lb)
        ub = np.array(ub)
        # print(lb, ub)
        return A, lb, ub

    def CalculateCurvature(self, xi_minus_1, xi, xi_plus_1, yi_minus_1, yi, yi_plus_1):
        vec_1 = np.array([xi_minus_1 - xi, yi_minus_1 - yi])
        vec_2 = np.array([xi_plus_1 - xi, yi_plus_1 - yi])
        if abs(np.cross(vec_1, vec_2)) < 1e-5:
            return 1e-5

        a = 1e-5 if abs(xi_minus_1 - xi) < 1e-5 else xi_minus_1 - xi
        b = 1e-5 if abs(xi - xi_plus_1) < 1e-5 else xi - xi_plus_1

        u = ((math.pow(xi_minus_1, 2) - math.pow(xi, 2) +
             math.pow(yi_minus_1, 2) - math.pow(yi, 2))/(2 * a))
        k_1 = (yi_minus_1 - yi)/a

        v = ((math.pow(xi, 2) - math.pow(xi_plus_1, 2) +
             math.pow(yi, 2) - math.pow(yi_plus_1, 2))/(2 * b))
        k_2 = (yi - yi_plus_1)/b

        Ry = (u - v) / (k_1 - k_2)
        Rx = v - k_2 * (u - v)/(k_1 - k_2)
        R = math.hypot(xi_minus_1 - Rx, yi_minus_1 - Ry)
        return 1.0/R

    def CalculatePathCurvature(self, x, y):
        path_curvature = []
        for i in range(1, len(x) - 2):
            path_curvature.append(self.CalculateCurvature(
                x[i-1], x[i], x[i+1], y[i-1], y[i], y[i+1]))
        return path_curvature

    def QPSolve(self):
        # Init solution
        Q = self.CalculateQ()
        P = self.CalculateP()
        A, lb, ub = self.CalculateAffineConstraint(self.x_ref_, self.y_ref_)

        if self.fixed_start_end_:
            lb[0] = self.x_ref_[0] - 1e-6
            ub[0] = self.x_ref_[0] + 1e-6
            lb[self.num_of_points_ - 1] = self.x_ref_[self.num_of_points_ - 1] - 1e-6
            ub[self.num_of_points_ - 1] = self.x_ref_[self.num_of_points_ - 1] + 1e-6

            lb[self.num_of_points_] = self.y_ref_[0] - 1e-6
            ub[self.num_of_points_] = self.y_ref_[0] + 1e-6
            lb[self.num_of_pos_variables -
                1] = self.y_ref_[self.num_of_points_ - 1] - 1e-6
            ub[self.num_of_pos_variables -
                1] = self.y_ref_[self.num_of_points_ - 1] + 1e-6

        prob = osqp.OSQP()
        prob.setup(Q, P, A, lb, ub, polish=True, eps_abs=1e-5, eps_rel=1e-5,
                   eps_prim_inf=1e-5, eps_dual_inf=1e-5)

        var_warm_start = np.array(
            self.x_ref_ + self.y_ref_ + [0.0 for n in range(self.num_of_slack_variables_)])
        prob.warm_start(var_warm_start)
        res = prob.solve()

        # Plot init solution
        plt.figure(num=1)
        plt.xlabel("x coordinate")
        plt.ylabel("y coordinate")
        legend_title = ['Ref', 'Iter_1']
        plt.plot(self.x_ref_, self.y_ref_, marker="o")
        plt.plot(res.x[0:self.num_of_points_],
                 res.x[self.num_of_points_:self.num_of_pos_variables],
                 marker="o")
        plt.title("Path SQP Smoother")

        ref_path_curvature = self.CalculatePathCurvature(
            self.x_ref_, self.y_ref_)
        print(ref_path_curvature)
        iter_1_path_curvature = self.CalculatePathCurvature(
            res.x[0:self.num_of_points_], res.x[self.num_of_points_:self.num_of_pos_variables])

        plt.figure(num=2)
        plt.ylabel("curvature")
        plt.title("Path curvature")
        plt.plot(ref_path_curvature, marker="o")
        plt.plot(iter_1_path_curvature, marker="o")

        last_obj_val = res.info.obj_val
        last_opt_res = res.x
        iter_num = 1
        pen_itr = 0
        while pen_itr < self.sqp_pen_max_iter_:
            sub_itr = 0
            while sub_itr < self.sqp_sub_max_iter_:
                # new iteration
                iter_num += 1
                Q = self.CalculateQ()
                P = self.CalculateP()
                A, lb, ub = self.CalculateAffineConstraint((last_opt_res[0:self.num_of_points_]).tolist(), (
                    last_opt_res[self.num_of_points_:self.num_of_pos_variables]).tolist())

                if self.fixed_start_end_:
                    lb[0] = self.x_ref_[0] - 1e-6
                    ub[0] = self.x_ref_[0] + 1e-6
                    lb[self.num_of_points_ -
                        1] = self.x_ref_[self.num_of_points_ - 1] - 1e-6
                    ub[self.num_of_points_ -
                        1] = self.x_ref_[self.num_of_points_ - 1] + 1e-6

                    lb[self.num_of_points_] = self.y_ref_[0] - 1e-6
                    ub[self.num_of_points_] = self.y_ref_[0] + 1e-6
                    lb[self.num_of_pos_variables -
                        1] = self.y_ref_[self.num_of_points_ - 1] - 1e-6
                    ub[self.num_of_pos_variables -
                        1] = self.y_ref_[self.num_of_points_ - 1] + 1e-6

                prob = osqp.OSQP()
                prob.setup(Q, P, A, lb, ub, polish=True, eps_abs=1e-5, eps_rel=1e-5,
                           eps_prim_inf=1e-5, eps_dual_inf=1e-5)

                prob.warm_start(last_opt_res)
                res = prob.solve()
                last_opt_res = res.x

                # plot new iteration's result
                plt.figure(num=1)
                plt.plot(res.x[0:self.num_of_points_],
                         res.x[self.num_of_points_:self.num_of_pos_variables], marker="o")
                legend_title.append("Iter_"+str(iter_num))

                iter_path_curvature = self.CalculatePathCurvature(
                    res.x[0:self.num_of_points_], res.x[self.num_of_points_:self.num_of_pos_variables])
                plt.figure(num=2)
                plt.plot(iter_path_curvature, marker="o")

                obj_val_tol = abs((res.info.obj_val -
                                  last_obj_val)/last_obj_val)
                last_obj_val = res.info.obj_val
                sub_itr += 1
                if obj_val_tol < self.sqp_ftol_:
                    break

            pen_itr += 1
            max_violation = self.CalculateConstraintViolation((last_opt_res[0:self.num_of_points_]).tolist(), (
                last_opt_res[self.num_of_points_:self.num_of_pos_variables]).tolist())
            print(max_violation)
            if max_violation < self.sqp_ctol_:
                break

        plt.figure(1)
        plt.legend(legend_title)
        plt.figure(2)
        plt.legend(legend_title)
        plt.show()
        return


def main():
    points_x = [1.0, 1.5, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.5, 2.9, 3.3, 3.8]
    points_y = [0.0, 0.3, 0.5, 0.9, 1.2, 1.5, 1.9, 2.2, 2.2, 2.2, 2.2, 2.2]
    lb = 0.3
    ub = 0.3
    smooth_weight = 4.0
    length_weight = 0.01
    distance_weight = 10.0
    slack_weight = 400
    curvature_limit = 0.2
    fixed_start_end = False

    sqp_smoother = SQPSmoother(
        points_x, points_y, lb, ub, curvature_limit, smooth_weight, length_weight, distance_weight, slack_weight, fixed_start_end)
    sqp_smoother.QPSolve()


if __name__ == "__main__":
    main()
