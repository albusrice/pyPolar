import numpy as np
from math import sqrt


class AWGN:
    def __init__(self, myPC, SNR):
        self.myPC = myPC
        self.EbNodB = SNR

        self.transmit()

    def transmit(self):
        rate = self.myPC.K / self.myPC.N
        EbNo = 10.0 ** (self.EbNodB / 10.0)
        sigma = sqrt(1.0 / (2.0 * rate * EbNo))

        self.llrs = 1 - 2.0 * self.myPC.u                      # BPSK bit symbol conversion
        self.llrs += sigma * np.random.randn(self.myPC.N)      # Add AWGN channel noise

        self.myPC.llrs = self.llrs
        # self.set_llr(self.llrs)

    # def set_llr(self, llr):
    #     # self.myPC.LLR[:, 0, :] = self.myPC.Ln
    #     self.myPC.LLR[:, 0, :] = np.tile(llr, (self.myPC.Ln, 1))


