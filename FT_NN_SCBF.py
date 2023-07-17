from NCBF_Synth import *

class FT_NN_SCBF(NCBF_Synth):
    def __init__(self, arch, act_layer, case, verbose=False):
        super().__init__(arch, act_layer, case, verbose=False)
        self.critic = NeuralCritic(case)
        self.veri = Verifier(NCBF=self, case=case,
                             grid_shape=[100, 100, 100],
                             verbose=verbose)

    def numerical_b_gamma(self, grad, gamma):
        # todo: debug
        return np.max(np.abs(grad)) * gamma

    def EKF(self):
        # todo: extended kalman filter gain for different sensor failure
        K = torch.ones([self.DIM, self.DIM])
        return K

    def SCBF_violations(self, model_output, delta_gamma,
                        feasibility_output, stochastic_term,
                        batch_length, rlambda):
        b_hat_gamma = (model_output - delta_gamma).transpose(0, 1)
        violations = -1 * feasibility_output - stochastic_term \
                     - torch.max(rlambda * torch.abs(b_hat_gamma), torch.zeros([1, batch_length]))
        return violations

    def SCBF_train(self, num_epoch):
        optimizer = optim.SGD(self.model.parameters(), lr=1e-3)
        scheduler = ExponentialLR(optimizer, gamma=0.9)
        # Generate data
        size = 40
        shape = []
        for _ in range(self.DIM):
            shape.append(size)
        rlambda = 1
        rdm_input = self.generate_data(size)
        # rdm_input = self.generate_input(shape)
        # ref_output = torch.unsqueeze(self.h_x(rdm_input.transpose(0, self.DIM)), self.DIM)
        ref_output = self.h_x(rdm_input.transpose(0, 1)).unsqueeze(1)
        batch_length = 4**self.DIM
        training_loader = DataLoader(list(zip(rdm_input, ref_output)), batch_size=batch_length, shuffle=True)
        pbar = tqdm(total=len(training_loader))
        veri_result = False
        gamma = 0.1
        for epoch in range(num_epoch):
            # Initialize loss
            running_loss = 0.0
            feasibility_running_loss = 0.0
            correctness_running_loss = 0.0
            trivial_running_loss = 0.0
            # Batch Training
            for X_batch, y_batch in training_loader:
                optimizer.zero_grad()
                model_output = self.forward(X_batch)

                warm_start_loss = self.warm_start(y_batch, model_output)
                correctness_loss = self.safe_correctness(y_batch, model_output, l_co=1, alpha1=1, alpha2=0)
                trivial_loss = self.trivial_panelty(ref_output, self.model.forward(rdm_input), 1)

                grad = self.numerical_gradient(X_batch, model_output, batch_length, epsilon=0.001)
                grad_vector = torch.vstack(grad)
                feasibility_output = self.feasibility_loss(grad_vector, X_batch)
                check_item = torch.max((-torch.abs(model_output)+0.1).reshape([1, batch_length]),
                                       torch.zeros([1, batch_length]))
                stochastic_term = -gamma * torch.linalg.norm(grad_vector @ self.EKF() * c)
                # feasibility_loss = torch.sum(torch.tanh(check_item*feasibility_output))

                # Our loss function
                # violations = -check_item * feasibility_output
                # Chuchu Fan loss function
                delta_gamma = self.numerical_b_gamma(grad_vector, gamma)
                violations = self.SCBF_violations(model_output, delta_gamma,
                                                  feasibility_output, stochastic_term,
                                                  batch_length, rlambda)
                # violations = -1 * feasibility_output - torch.max(rlambda * torch.abs(model_output.transpose(0, 1)),
                #                                                  torch.zeros([1, batch_length]))
                feasibility_loss = 100 * torch.sum(torch.max(violations - 1e-4, torch.zeros([1, batch_length])))
                loss = self.def_loss(1*correctness_loss + 1*feasibility_loss + 1*trivial_loss)
                loss.backward()
                optimizer.step()

                # Print Detailed Loss
                running_loss += loss.item()
                feasibility_running_loss += feasibility_loss.item()
                correctness_running_loss += correctness_loss.item()
                trivial_running_loss += trivial_loss.item()
                # Process Bar Print Losses
                pbar.set_postfix({'Loss': running_loss,
                                  'Floss': feasibility_running_loss,
                                  'Closs': correctness_running_loss,
                                  'Tloss': trivial_running_loss,
                                  'PVeri': str(veri_result)})
                pbar.update(1)

            scheduler.step()
            veri_result, num = self.veri.proceed_verification()
            pbar.reset()

ObsAvoid = ObsAvoid()
newCBF = FT_NN_SCBF([32, 32], [True, True], ObsAvoid, verbose=False)
# newCBF.veri.proceed_verification()
for restart in range(3):
    newCBF.SCBF_train(5)
