import autograd.numpy as np

class PhaseSpace:
    def __init__(self, F_fun, G_fun, x0, DIM):
        self.DIM = DIM
        self.x0 = x0

        self.F = F_fun
        self.G = G_fun
        self.W = [self.W_state, self.W_gate, self.W_meas]

    def W_state(self, state, x):
        DIM = self.DIM
        F1q = self.F(x)
        return np.real(np.einsum('ijkl,lk->ij', F1q, state))

    def W_gate(self, gate, x_in_list, x_out_list):
        DIM = self.DIM
        if len(x_in_list)==1:
            G_in = self.G(x_in_list[0])
            F_out = self.F(x_out_list[0])
            U_ev = np.einsum('lk,ijkn,mn->ijlm', gate, G_in, gate.conj())
            return np.real(np.einsum('ijkl,mnlk->ijmn', U_ev, F_out))

        if len(x_in_list)==2:
            G_in1 = self.G(x_in_list[0])
            G_in2 = self.G(x_in_list[1])

            F_out1 = self.F(x_out_list[0])
            F_out2 = self.F(x_out_list[1])

            G_in = np.einsum('ijkl,mnrs->ijmnkrls',
                     G_in1, G_in2).reshape(
                     (DIM,DIM,DIM,DIM,DIM*DIM,DIM*DIM))
            F_out = np.einsum('ijkl,mnrs->ijmnkrls',
                      F_out1, F_out2).reshape(
                      (DIM,DIM,DIM,DIM,DIM*DIM,DIM*DIM))
            U_ev = np.einsum('lk,ijsrkn,mn->ijsrlm',
                             gate, G_in, gate.conj())
            return np.real(np.einsum('ijsrkl,mnablk->ijsrmnab',U_ev, F_out))

        if len(x_in_list)==3:
            G_in1 = self.G(x_in_list[0])
            G_in2 = self.G(x_in_list[1])
            G_in3 = self.G(x_in_list[2])

            F_out1 = self.F(x_out_list[0])
            F_out2 = self.F(x_out_list[1])
            F_out3 = self.F(x_out_list[2])

            G_in = np.einsum('ijkl,mnrs,xyzw->ijmnxykrzlsw',
                     G_in1, G_in2, G_in3).reshape(
                     (DIM,DIM,DIM,DIM,DIM,DIM,DIM*DIM*DIM,DIM*DIM*DIM))
            F_out = np.einsum('ijkl,mnrs,xyzw->ijmnxykrzlsw',
                      F_out1, F_out2, F_out3).reshape(
                      (DIM,DIM,DIM,DIM,DIM,DIM,DIM*DIM*DIM,DIM*DIM*DIM))
            U_ev = np.einsum('lk,ijsrxykn,mn->ijsrxylm',
                             gate, G_in, gate.conj())
            return np.real(np.einsum('ijsrxykl,mnabzwlk->ijsrxymnabzw',
                                     U_ev, F_out))

        if len(x_in_list)==4:
            G_in1 = self.G(x_in_list[0])
            G_in2 = self.G(x_in_list[1])
            G_in3 = self.G(x_in_list[2])
            G_in4 = self.G(x_in_list[3])

            F_out1 = self.F(x_out_list[0])
            F_out2 = self.F(x_out_list[1])
            F_out3 = self.F(x_out_list[2])
            F_out4 = self.F(x_out_list[3])

            G_in = np.einsum('ijkl,mnrs,xyzw,abcd->ijmnxyabkrzclswd',
                     G_in1, G_in2, G_in3, G_in4).reshape(
                     (DIM,DIM,DIM,DIM,DIM,DIM,DIM,DIM,DIM**4,DIM**4))
            F_out = np.einsum('ijkl,mnrs,xyzw,abcd->ijmnxyabkrzclswd',
                      F_out1,F_out2,F_out3,F_out4).reshape(
                      (DIM,DIM,DIM,DIM,DIM,DIM,DIM,DIM,DIM**4,DIM**4))
            U_ev = np.einsum('lk,ijsrxyabkn,mn->ijsrxyablm',
                             gate, G_in, gate.conj())
            return np.real(np.einsum(
                           'ijsrxycdkl,mnabzweflk->ijsrxycdmnabzwef',
                           U_ev, F_out))

        if len(x_in_list)==5:
            G_in1 = self.G(x_in_list[0])
            G_in2 = self.G(x_in_list[1])
            G_in3 = self.G(x_in_list[2])
            G_in4 = self.G(x_in_list[3])
            G_in5 = self.G(x_in_list[4])

            F_out1 = self.F(x_out_list[0])
            F_out2 = self.F(x_out_list[1])
            F_out3 = self.F(x_out_list[2])
            F_out4 = self.F(x_out_list[3])
            F_out5 = self.F(x_out_list[4])

            G_in = np.einsum('abkp,cdlq,efmr,ghns,ijot->abcdefghijklmnopqrst',
                     G_in1,G_in2,G_in3,G_in4,G_in5).reshape(
                     (DIM,DIM,DIM,DIM,DIM,DIM,DIM,DIM,DIM,DIM,DIM**5,DIM**5))
            F_out = np.einsum('abkp,cdlq,efmr,ghns,ijot->abcdefghijklmnopqrst',
                      F_out1,F_out2,F_out3,F_out4,F_out5).reshape(
                      (DIM,DIM,DIM,DIM,DIM,DIM,DIM,DIM,DIM,DIM,DIM**5,DIM**5))
            U_ev = np.einsum('ml,abcdefghijlk,nk->abcdefghijmn',
                             gate, G_in, gate.conj())
            return np.real(np.einsum(
                           'abcdefghijuv,klmnopqrstvu->abcdefghijklmnopqrst',
                           U_ev, F_out))

    def W_meas(self, meas, x):
        G1q = self.G(x)
        return np.real(np.einsum('ijkl,lk->ij', G1q, meas))