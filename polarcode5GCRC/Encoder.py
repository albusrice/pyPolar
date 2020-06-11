import numpy as np


class Encoder:
    def __init__(self, myPC):
        self.myPC = myPC
        self.polar_encode()

    def polar_encode(self):
        num_elements = 1
        index = None
        for i in range(self.myPC.depth):

            if self.myPC.N <= 512:
                index = np.array(([True] * num_elements + [False] * num_elements) * (self.myPC.N // 2 // num_elements))
            else:
                # most efficient for very large array
                index = np.tile([True] * num_elements + [False] * num_elements, self.myPC.N // 2 // num_elements)

            self.myPC.u[index] = np.logical_xor(self.myPC.u[index], self.myPC.u[np.logical_not(index)])
            num_elements *= 2
