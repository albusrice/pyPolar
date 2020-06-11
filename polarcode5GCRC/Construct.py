import numpy as np
from math import *
from polarcode5GCRC.utils import *


class Construct:
    def __init__(self, N, K,
                 num_decoders=1,
                 crc=False,
                 crc_polynomial=np.array([1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1]) # x ** 11 + x ** 10 + x ** 9 + x ** 5 + 1
                 ):
        """
         Attributes
        ----------
        :param N: int
            polar code length
        :param K: int
            message block length
        """
        self.reliability_sequence_full = full_reliability_sequence_5g()

        self.crc_flag = crc
        self.crc_polynomial = crc_polynomial
        self.crcl = len(self.crc_polynomial) - 1
        self.Ln = num_decoders
        print(num_decoders)

        self.N = N
        self.K = K
        self.depth = int(log2(N))

        # Encoding structure construction
        self.reliability_sequence = self.reliability_sequence_full[self.reliability_sequence_full < self.N]
        self.frozen_bits = self.reliability_sequence[:self.N - self.K - self.crcl * self.crc_flag]
        self.msg_bits = self.reliability_sequence[len(self.frozen_bits):]

        self.message_sent = None                                 # message
        self.x = np.zeros(self.N, dtype=int)                    # encoder message vector
        self.u = np.zeros(self.N, dtype=int)                    # encoded message
        self.llrs = np.zeros(self.N)                            # message llrs
        self.message_received = np.zeros(self.N, dtype=int)     # decoded message

        # Decoding structure construction (build a complete binary tree)
        self.total_nodes = 2 ** (self.depth + 1) - 1
        self.decoding_tree = build_binary_tree(self.N, self.depth)

        self.LLR = np.zeros(shape=[self.Ln, self.depth + 1, self.N], dtype=float)
        self.hat_u = np.zeros(shape=[self.Ln, self.depth + 1, self.N], dtype=float)
        self.PML = np.infty * np.ones(2 * self.Ln)
        self.PML[0] = 0

    def set_message(self, msg):
        if len(msg) == self.K:
            # print(self.msg_bits)
            self.message_sent = msg
            if self.crc_flag:
                temp_K = self.K + self.crcl
                msgcrc = np.zeros(temp_K)
                msgcrc[:self.K] = msg
                _, rem = gfdeconv(msgcrc, self.crc_polynomial)
                msgcrc[temp_K - len(rem):] = rem
                self.x[self.msg_bits] = msgcrc
                self.u[self.msg_bits] = msgcrc

            else:
                self.x[self.msg_bits] = msg
                self.u[self.msg_bits] = msg
        else:
            raise Exception("Input message length ({}) not equal to {}".format(len(msg), self.K))

    def __str__(self):
        string = "========== Polar Codes ========== \n"
        string += "N : {} \n".format(self.N)
        string += "K : {} \n".format(self.K)
        string += "Depth : {} \n\n".format(self.depth)
        string += "Ordered bits: \n"
        string += format_list(self.reliability_sequence)
        string += "Frozen bits (Least to most reliable): \n"
        string += format_list(self.frozen_bits)
        string += "Message :\n"
        string += format_list(self.x)
        string += "=" * 33 + "\n"
        # self.reliability_sequence_full.sort()
        # string += format_list(self.reliability_sequence_full)
        return string

