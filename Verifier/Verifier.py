from Verifier.ReLUNN_Decom import *
from Cases.Case import *
import time
import warnings
warnings.filterwarnings("ignore")

class Verifier(ReLUNN_Decom):
    def __init__(self, NCBF, case, grid_shape, verbose=True):
        super().__init__(NCBF, grid_shape, verbose)
        self.case = case
        self.dbdxg = 0
        self.x0 = np.zeros([self.DIM, 1])
        self.num_neuron = self.NN.arch[0]

    def dbdxf(self, x, W_overl):
        fx = self.case.f_x(torch.Tensor(x).reshape([1, self.DIM])).numpy()
        dbdxf = W_overl @ fx
        return dbdxf

    def y_dbdxf(self, xy, W_overl_inter):
        sum_range = int(len(W_overl_inter) / 1)
        y = xy[-2 * sum_range:]
        x = xy[:-2 * sum_range]
        W_overl = W_overl_inter[0]
        xarray = self.dbdxf(x, W_overl)
        for i in range(sum_range):
            W_overl = W_overl_inter[i * 1:(i + 1) * 1][0]
            if i == 0:
                xarray = self.dbdxf(x, W_overl)
            else:
                xarray = np.vstack([xarray, self.dbdxf(x, W_overl)])
        obj_sum = np.vstack([xarray, xarray])
        return obj_sum.transpose() @ y


    def verification(self, actual_set):
        problematic_set = []
        for act_array in actual_set:
            W_overl, r_overl, B_act, B_inact = self.activated_weight_bias_ml(torch.Tensor(act_array), self.num_neuron)
            # compute boundary condition of polyhedron
            W_overl = W_overl.numpy()
            r_overl = r_overl.numpy()
            W_Bound = np.array(-B_act[0] + B_inact[0])
            r_Bound = np.array(-B_act[1] - B_inact[1])

            if self.dbdxg == 0:
                lcon = LinearConstraint(W_Bound, -np.inf * np.ones(len(W_Bound)), -r_Bound.reshape(len(r_Bound)))
                eqcon = LinearConstraint(W_overl[0], -r_overl[0], -r_overl[0])
                res = minimize(self.dbdxf, self.x0, args=W_overl[0], constraints=[lcon, eqcon], tol=1e-6)
                # print(res.fun)
                if res.fun < 0:
                    problematic_set.append(act_array.copy())

        # print(len(problematic_set))
        if len(problematic_set) == 0:
            return True
        else:
            return False

    def inter_verification(self, actual_set):
        problematic_set = []

        for idx in range(len(actual_set)):
            act_array = actual_set[idx]
            W_overl, r_overl, B_act, B_inact = self.activated_weight_bias_ml(torch.Tensor(act_array), self.num_neuron)
            # compute boundary condition of polyhedron
            W_overl = W_overl.numpy()
            r_overl = r_overl.numpy()
            W_Bound = np.array(-B_act[0] + B_inact[0])
            r_Bound = np.array(-B_act[1] - B_inact[1])
            if idx == 0:
                W_overl_inter = W_overl
                r_overl_inter = r_overl
                W_Bound_inter = W_Bound
                r_Bound_inter = r_Bound
            else:
                W_overl_inter = np.vstack([W_overl_inter, W_overl])
                r_overl_inter = np.vstack([r_overl_inter, r_overl])
                W_Bound_inter = np.vstack([W_Bound_inter, W_Bound])
                r_Bound_inter = np.vstack([r_Bound_inter, r_Bound])

            # con = lambda x: W_overl[0]*(x[1] + 2 * x[0] * x[1]) + W_overl[1]*(-x[0] + 2 * x[0] ** 2 - x[1] ** 2)
            # nlc = NonlinearConstraint(con, -np.inf*np.ones(len(W_Bound)), -r_Bound)
            lcon = LinearConstraint(W_Bound, -np.inf * np.ones(len(W_Bound)), -r_Bound.reshape(len(r_Bound)))
            eqcon = LinearConstraint(W_overl[0], -r_overl[0], -r_overl[0])
            res = minimize(self.dbdxf, self.x0, args=W_overl[0], constraints=[lcon, eqcon], tol=1e-6)

            if res.fun < 0:
                problematic_set.append(act_array.copy())
        # print(len(problematic_set)/len(actual_set))
        if len(problematic_set) == 0:
            return True
        else:
            size = [len(W_overl), len(r_overl[0]), len(W_Bound), len(r_Bound)]
            res_value = self.suf_nec_inter_verification(W_overl_inter, r_overl_inter,
                                                        W_Bound_inter, r_Bound_inter, size)
            if res_value >= 0:
                return True
            else:
                return False

    def suf_nec_inter_verification(self, W_overl_inter, r_overl_inter, W_Bound_inter, r_Bound_inter, size):
        results = []
        size_W_overl = size[0]
        size_r_overl = size[1]
        size_W_Bound = size[2]
        size_r_Bound = size[3]
        sum_range = int(len(W_overl_inter) / size_W_overl)
        for i in range(sum_range):
            W_overl = W_overl_inter[i * size_W_overl:(i + 1) * size_W_overl]
            r_overl = r_overl_inter[i * size_r_overl:(i + 1) * size_r_overl]
            W_Bound = W_Bound_inter[i * size_W_Bound:(i + 1) * size_W_Bound]
            r_Bound = r_Bound_inter[i * size_r_Bound:(i + 1) * size_r_Bound]

            lcon = LinearConstraint(W_Bound, -np.inf * np.ones(len(W_Bound)), -r_Bound.reshape(len(r_Bound)))
            eqcon = LinearConstraint(W_overl[0], -r_overl[0], -r_overl[0])
            res = minimize(self.dbdxf, self.x0, args=W_overl[0], constraints=[lcon, eqcon], tol=1e-6)
            results.append(res.fun)

        results_array = np.asarray(results)
        initial_x0 = self.x0
        for num in range(sum_range - 1):
            initial_x0 = np.vstack([initial_x0, self.x0])
        y0 = np.zeros(2 * sum_range)
        initial_state = np.vstack([initial_x0, y0.reshape([y0.shape[0], 1])])
        xy0 = np.vstack([self.x0, y0.reshape([y0.shape[0], 1])])
        # np.vstack([W_Bound_inter, np.zeros([2 * sum_range, W_Bound_inter.shape[1]])])
        # np.vstack([np.zeros([W_Bound_inter.shape[0], 2 * sum_range]), np.eye(2 * sum_range)])
        lcon = LinearConstraint(
            np.hstack([np.vstack([W_Bound_inter, np.zeros([2 * sum_range, W_Bound_inter.shape[1]])]),
                       np.vstack([np.zeros([W_Bound_inter.shape[0], 2 * sum_range]), np.eye(2 * sum_range)])]),
            np.hstack([-np.inf * np.ones(len(r_Bound_inter)), np.zeros(2 * sum_range)]),
            np.hstack([-r_Bound_inter.reshape(len(r_Bound_inter)), np.inf * np.ones(2 * sum_range)]))

        eqcon = LinearConstraint(
            np.hstack([np.vstack([W_overl_inter, np.zeros([2 * sum_range, W_overl_inter.shape[1]])]),
                       np.vstack(
                           [results_array.reshape([sum_range, 1]) * np.ones([W_overl_inter.shape[0], 2 * sum_range]),
                            np.eye(2 * sum_range)])]),
            np.hstack([-r_overl_inter[:, 0], np.zeros(2 * sum_range)]),
            np.hstack([-r_overl_inter[:, 0], np.zeros(2 * sum_range)]))
        res = minimize(self.y_dbdxf, xy0.reshape(len(xy0)), args=W_overl_inter, constraints=[lcon, eqcon], tol=1e-6)
        return res.fun

    def actual_set_check(self, unique_set):
        res_list = []
        U_actset_list = []
        pts = []
        actual_set_list = []
        for act_array in unique_set:
            act_str = np.array2string(act_array.reshape([len(act_array)]))
            W_overl, r_overl, B_act, B_inact = self.activated_weight_bias_ml(torch.Tensor(act_array), self.num_neuron)
            # compute boundary condition of polyhedron
            W_Bound = torch.Tensor(-B_act[0] + B_inact[0])
            r_Bound = torch.Tensor(-B_act[1] - B_inact[1])
            res_zero = linprog(c=[1, 1],
                               A_ub=W_Bound, b_ub=-r_Bound,
                               A_eq=W_overl, b_eq=-r_overl, bounds=tuple(self.DOMAIN),
                               method='highs')
            # if self.verbose:
            #     print(res_zero.success)
            res_list.append(res_zero.success)
            if res_zero.success:
                actual_set_list.append(act_array.copy())

        # U_actset = set(U_actset_list)
        # if self.verbose:
        #     print('U_actset', len(set(U_actset)))
        #     print('Activation Patterns', sum(res_list))
        return actual_set_list

    def proceed_verification(self):
        if self.verbose:
            print('-------------- Verification Start --------------')
        t_start = time.time()
        act_sets_list, activated_sets,  possible_intersections = self.preceed_decompose()
        self.act_sets_list = act_sets_list
        self.activated_sets = activated_sets
        self.possible_intersections = possible_intersections
        act_sets_array = np.asarray(self.act_sets_list)
        unique_set = np.unique(act_sets_array, axis=0)

        actual_set_list = self.actual_set_check(unique_set)
        self.actual_set_list = actual_set_list
        intersections, act_intersections_list = self.find_intersects(self.actual_set_list, self.possible_intersections)
        self.intersections = intersections
        self.act_intersections_list = act_intersections_list

        t_end = time.time()
        time_spent = t_end - t_start
        if self.verbose:
            print('-------------Decomposition Complete-------------')
            print('Unique set', len(np.unique(act_sets_array, axis=0)))
            print('Intersections', len(intersections))
            print('activated set compute complete with size', len(activated_sets))
            print('Decomposition finished in', time_spent, 'seconds')

        # Verification
        veri_res_set = self.verification(actual_set_list)
        veri_res_intersect = []
        for act_intersections in act_intersections_list:
            veri_res_intersect_item = self.inter_verification(act_intersections)
            veri_res_intersect.append(veri_res_intersect_item)
        if self.verbose:
            print('res_set', veri_res_set)
            print('res_bd_set', veri_res_intersect)
            v_end = time.time()
            veri_time = v_end - t_start
            print('Verification finished in', veri_time)
            print('-------------Verification  Complete-------------')
        if veri_res_set and all(veri_res_intersect):
            if self.verbose:
                print('Successfully Verified')
            return True, len(actual_set_list)
        else:
            if self.verbose:
                print('Failed Verification')
            return False, len(actual_set_list)

