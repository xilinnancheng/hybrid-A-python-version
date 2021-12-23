import cyipopt
import math
import numpy as np
from scipy import sparse
from matplotlib import pyplot as plt


class IpoptSmootherProblem:
    def __init__(self, Q, P, num_of_points):
        self.Q = Q
        self.P = P
        self.num_of_points = num_of_points
        self.num_of_pos_variables = 2 * num_of_points
        self.num_of_slack_variables = num_of_points - 2
        self.num_of_variables = self.num_of_pos_variables + self.num_of_slack_variables

    def objective(self, x):
        print(x)
        print(np.dot(np.dot(x.transpose(), self.Q), x) + np.dot(self.P, x))
        return np.dot(np.dot(x.transpose(), self.Q), x) + np.dot(self.P, x)

    def gradient(self, x):
        return 2.0 * np.dot(self.Q, x) + self.P

    def CalculateSmooth(self, xi_minus_1, xi, xi_plus_1, yi_minus_1, yi, yi_plus_1):
        return math.pow(xi_minus_1 + xi_plus_1 - 2 * xi, 2.0) + \
            math.pow(yi_minus_1 + yi_plus_1 - 2 * yi, 2.0)

    def constraints(self, x):
        g = []
        for i in range(1, self.num_of_points - 1):
            x_index = i
            y_index = i + self.num_of_points
            slack_index = self.num_of_pos_variables + i - 1
            g.append(self.CalculateSmooth(x[x_index-1], x[x_index], x[x_index+1], x[y_index-1], x[y_index], x[y_index+1]) -
                     x[slack_index])
        return np.array(g)

    def jacobian(self, x):
        row = []
        col = []
        data = []

        row_index = 0
        for i in range(1, self.num_of_points - 1):
            xi = i
            yi = i + self.num_of_points
            row.append(row_index)
            col.append(xi - 1)
            data.append(2.0 * (x[xi-1] + x[xi+1] - 2.0 * x[xi]))

            row.append(row_index)
            col.append(i)
            data.append(-4.0 * (x[xi-1] + x[xi+1] - 2.0 * x[xi]))

            row.append(row_index)
            col.append(i + 1)
            data.append(2.0 * (x[xi-1] + x[xi+1] - 2.0 * x[xi]))

            row.append(row_index)
            col.append(yi - 1)
            data.append(2.0 * (x[yi-1] + x[yi+1] - 2.0 * x[yi]))

            row.append(row_index)
            col.append(yi)
            data.append(-4.0 * (x[yi-1] + x[yi+1] - 2.0 * x[yi]))

            row.append(row_index)
            col.append(yi + 1)
            data.append(2.0 * (x[yi-1] + x[yi+1] - 2.0 * x[yi]))

            row.append(row_index)
            col.append(self.num_of_pos_variables + i - 1)
            data.append(-1.0)

            row_index += 1

        return (sparse.csc_matrix((data, (row, col)), shape=(
            self.num_of_slack_variables, self.num_of_variables))).toarray()

    def hessianstructure(self):
        return np.nonzero(np.tril(np.ones((self.num_of_variables, self.num_of_variables))))

    def hessian(self, x, lagrange, obj_factor):
        hessian_f = self.Q
        hessian_g = np.zeros((self.num_of_variables,
                             self.num_of_variables))
        for i in range(1, self.num_of_points - 1):
            row = []
            col = []
            data = []

            row.append(i-1)
            col.append(i-1)
            data.append(2)

            row.append(i-1)
            col.append(i)
            data.append(-4)

            row.append(i-1)
            col.append(i+1)
            data.append(2)

            row.append(self.num_of_points + i-1)
            col.append(self.num_of_points + i-1)
            data.append(2)

            row.append(self.num_of_points + i-1)
            col.append(self.num_of_points + i)
            data.append(-4)

            row.append(self.num_of_points + i-1)
            col.append(self.num_of_points + i+1)
            data.append(2)

            row.append(i)
            col.append(i-1)
            data.append(-4)

            row.append(i)
            col.append(i)
            data.append(8)

            row.append(i)
            col.append(i+1)
            data.append(-4)

            row.append(self.num_of_points + i)
            col.append(self.num_of_points + i-1)
            data.append(-4)

            row.append(self.num_of_points + i)
            col.append(self.num_of_points + i)
            data.append(8)

            row.append(self.num_of_points + i)
            col.append(self.num_of_points + i+1)
            data.append(-4)

            row.append(i+1)
            col.append(i-1)
            data.append(2)

            row.append(i+1)
            col.append(i)
            data.append(-4)

            row.append(i+1)
            col.append(i+1)
            data.append(2)

            row.append(self.num_of_points + i+1)
            col.append(self.num_of_points + i-1)
            data.append(2)

            row.append(self.num_of_points + i+1)
            col.append(self.num_of_points + i)
            data.append(-4)

            row.append(self.num_of_points + i+1)
            col.append(self.num_of_points + i+1)
            data.append(2)

            hessian_g += lagrange[i-1] * (sparse.csc_matrix((data, (row, col)), shape=(
                self.num_of_variables, self.num_of_variables))).toarray()
        row_tmp, col_tmp = self.hessianstructure()
        H = 2.0 * obj_factor * hessian_f + hessian_g
        # print(H[row_tmp, col_tmp])
        # print(2.0 * obj_factor * hessian_f)
        return H[row_tmp, col_tmp]

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu, d_norm, regularization_size, alpha_du, alpha_pr, ls_trials):
        print("Objective value at iteration #%d is - %g" %
              (iter_count, obj_value))

# f(x) = 1/2 * x^T * Q * x + p^T * x


class IpoptSmoother:
    def __init__(self, points_x, points_y, lb, ub, curvature_limit, smooth_weight, length_weight, distance_weight, slack_weight, fixed_start_end, enable_plot):
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
        self.lb_ = []
        self.ub_ = []
        self.x_optimized = []
        self.y_optimized = []
        self.curvature_limit_ = curvature_limit
        self.fixed_start_end_ = fixed_start_end
        self.enable_plot_ = enable_plot

        total_length = 0.0
        for i in range(self.num_of_points_ - 1):
            total_length += math.hypot(self.x_ref_[i+1] - self.x_ref_[i],
                                       self.y_ref_[i+1] - self.y_ref_[i])
        self.average_length = total_length / (self.num_of_points_ - 1)
        # print(self.average_length)

        for i in range(self.num_of_points_):
            if self.fixed_start_end_ & (i == 0 or i == self.num_of_points_ - 1):
                self.lb_.append(self.x_ref_[i])
                self.ub_.append(self.x_ref_[i])
            else:
                self.lb_.append(self.x_ref_[i] - lb)
                self.ub_.append(self.x_ref_[i] + ub)
        for i in range(self.num_of_points_):
            if self.fixed_start_end_ & (i == 0 or i == self.num_of_points_ - 1):
                self.lb_.append(self.y_ref_[i])
                self.ub_.append(self.y_ref_[i])
            else:
                self.lb_.append(self.y_ref_[i] - lb)
                self.ub_.append(self.y_ref_[i] + ub)
        for i in range(self.num_of_slack_variables_):
            self.lb_.append(0.0)
            self.ub_.append(1e10)

        self.Q = self.CalculateQ()
        self.P = self.CalculateP()

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
            if row[i] == col[i]:
                data[i] *= 0.5

        Q = sparse.csc_matrix((data, (row, col)), shape=(
            self.num_of_variables_, self.num_of_variables_))
        Q += Q.transpose()
        # print(Q.toarray())
        return Q.toarray()

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
        for i in range(1, len(x) - 1):
            path_curvature.append(self.CalculateCurvature(
                x[i-1], x[i], x[i+1], y[i-1], y[i], y[i+1]))
        return path_curvature

    def Solve(self):
        if self.num_of_variables_ < 6:
            self.x_optimized = self.x_ref_
            self.y_optimized = self.y_ref_
            return

        problem = IpoptSmootherProblem(self.Q, self.P, self.num_of_points_)
        # problem.objective(np.array([1.0, 1.5, 2.1, 0.0, 0.3, 0.5, 0.0]))
        curvature_constraint = pow(
            pow(self.average_length, 2.0) * self.curvature_limit_, 2.0)

        cl = np.ones(self.num_of_slack_variables_) * -1e10
        cu = np.ones(self.num_of_slack_variables_) * curvature_constraint

        x_init = np.array(self.x_ref_ + self.y_ref_ +
                          [0.0 for n in range(self.num_of_slack_variables_)])

        nlp = cyipopt.Problem(self.num_of_variables_,
                              self.num_of_slack_variables_,
                              problem_obj=problem,
                              lb=self.lb_,
                              ub=self.ub_,
                              cl=cl,
                              cu=cu)

        nlp.add_option('mu_strategy', 'adaptive')
        nlp.add_option('tol', 1e-5)
        x, info = nlp.solve(x_init)

        self.x_optimized = x[0:self.num_of_points_]
        self.y_optimized = x[self.num_of_points_:self.num_of_pos_variables]
        if self.enable_plot_:
            plt.figure(num=1)
            plt.xlabel("x coordinate")
            plt.ylabel("y coordinate")
            legend_title = ['Ref', 'Iter_1']
            plt.plot(self.x_ref_, self.y_ref_, marker="o")
            plt.plot(self.x_optimized,
                     self.y_optimized,
                     marker="o")
            plt.title("Path Ipopt Smoother")

            ref_path_curvature = self.CalculatePathCurvature(
                self.x_ref_, self.y_ref_)
            iter_1_path_curvature = self.CalculatePathCurvature(
                self.x_optimized, self.y_optimized)

            plt.figure(num=2)
            plt.ylabel("curvature")
            plt.title("Path curvature")
            plt.plot(ref_path_curvature, marker="o")
            plt.plot(iter_1_path_curvature, marker="o")
            plt.show()


def main():
    points_x = [1.0, 1.5, 2.1, 2.1, 2.1, 2.1, 2.1, 2.1, 2.5, 2.9, 3.3, 3.8]
    points_y = [0.0, 0.3, 0.5, 0.9, 1.2, 1.5, 1.9, 2.2, 2.2, 2.2, 2.2, 2.2]
    # points_x = [1.0, 1.5, 2.1, 2.1]
    # points_y = [0.0, 0.3, 0.5, 1.2]
    lb = 0.3
    ub = 0.3
    smooth_weight = 1000.0
    length_weight = 2.0
    distance_weight = 1.0
    slack_weight = 1000.0
    curvature_limit = 0.2
    fixed_start_end = True
    enable_plot = True

    ipopt_smoother = IpoptSmoother(
        points_x, points_y, lb, ub, curvature_limit, smooth_weight, length_weight, distance_weight, slack_weight, fixed_start_end, enable_plot)
    ipopt_smoother.Solve()


if __name__ == "__main__":
    main()
