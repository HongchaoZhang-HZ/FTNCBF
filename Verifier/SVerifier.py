from Verifier.Verifier import *

class Stochastic_Verifier(Verifier):
    '''
    Define verifier for stochastic NCBFs
    '''
    def __init__(self, NCBF, EKFGain, case, grid_shape, verbose=True):
        super().__init__(NCBF, case, grid_shape, verbose=verbose)
        # SNCBF use EKF estimator
        # self.EKF = EKF
        self.EKFGain = EKFGain.numpy()
        # self.delta_gamma = delta_gamma
        self.gamma = 0.1
        self.c = torch.diag(torch.ones(self.DIM)).numpy()
        # self.b_gamma = self.NN - self.gamma * self.delta_gamma
        self.W_max = 0
        self.W_list = []
        self.r_list = []
        self.WB_list = []
        self.rB_list = []


    def LinearExpression(self, actual_set):
        for act_array in actual_set:
            W_overl, r_overl, B_act, B_inact = self.activated_weight_bias_ml(torch.Tensor(act_array), self.num_neuron)
            # compute boundary condition of polyhedron
            self.W_list.append(W_overl.numpy())
            self.r_list.append(r_overl.numpy())
            self.WB_list.append(np.array(-B_act[0] + B_inact[0]))
            self.rB_list.append(np.array(-B_act[1] - B_inact[1]))
            self.W_max = np.maximum(self.W_max, np.linalg.norm(W_overl))

    def dbdxf(self, x, W_overl):
        # stochastic version
        fx = self.case.f_x(torch.Tensor(x).reshape([1, self.DIM])).numpy()
        dbdxf = W_overl @ fx

        EKF_term = W_overl @ self.EKFGain @ self.c
        stochastic_term = self.gamma * np.linalg.norm(EKF_term) - self.W_max*self.gamma

        return dbdxf - stochastic_term

    def verification(self, actual_set):
        problematic_set = []
        self.LinearExpression(actual_set)
        for i in range(len(actual_set)):
            W_overl = self.W_list[i]
            r_overl = self.r_list[i]
            W_Bound = self.WB_list[i]
            r_Bound = self.rB_list[i]

            if self.dbdxg == 0:
                lcon = LinearConstraint(W_Bound, -np.inf * np.ones(len(W_Bound)), -r_Bound.reshape(len(r_Bound)))
                eqcon = LinearConstraint(W_overl[0], -r_overl[0], -r_overl[0])
                res = minimize(self.dbdxf, self.x0, args=W_overl[0], constraints=[lcon, eqcon], tol=1e-6)
                # print(res.fun)
                if res.fun < 0:
                    problematic_set.append(actual_set[i].copy())

        # print(len(problematic_set))
        if len(problematic_set) == 0:
            return True
        else:
            return False

    def proceed_verification(self):
        if self.verbose:
            print('-------------- Verification Start --------------')
        t_start = time.time()
        act_sets_list, activated_sets, possible_intersections = self.preceed_decompose()
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
        if self.verbose:
            print('veri_res_set', veri_res_set)
        if not veri_res_set:
            if self.verbose:
                print('Failed Verification')
            return False, len(actual_set_list)
        veri_res_intersect = []
        for i in range(len(act_intersections_list)):
            veri_res_intersect_item = self.inter_verification(act_intersections_list[i])
            # veri_res_intersect.append(veri_res_intersect_item)
            print(i, veri_res_intersect_item, len(act_intersections_list[i]))
            if not veri_res_intersect_item:
                if self.verbose:
                    print('Failed Verification')
                return False, len(actual_set_list)
            # else:
            #     # Todo: This branch need to be deleted when the memory issue is fixed
            #     if self.verbose:
            #         print('Successfully Verified')
            #     return True, len(actual_set_list)

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