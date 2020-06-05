import numpy as np
from math import *
from polarcodes5G.utils import format_list, full_reliability_sequence_5g, BTreeNode


class Construct:
    def __init__(self, N, K):
        """
         Attributes
        ----------
        :param N: int
            polar code length
        :param K: int
            message block length
        """
        self.reliability_sequence_full = full_reliability_sequence_5g()

        self.N = N
        self.K = K
        self.depth = int(log2(N))

        # Encoding structure construction
        self.reliability_sequence = self.reliability_sequence_full[self.reliability_sequence_full < self.N]
        self.frozen_bits = self.reliability_sequence[:self.N - self.K]
        self.msg_bits = self.reliability_sequence[self.N - self.K:]

        self.message_sent = None                                 # message
        self.x = np.zeros(self.N, dtype=int)                    # encoder message vector
        self.u = np.zeros(self.N, dtype=int)                    # encoded message
        self.llrs = np.zeros(self.N)                            # message llrs
        self.message_received = np.zeros(self.N, dtype=int)     # decoded message

        # Decoding structure construction (build a complete binary tree)
        self.total_nodes = 2 ** (self.depth + 1) - 1
        self.decoding_tree = [BTreeNode(i) for i in range(self.total_nodes)]

        self.decoding_tree[0].L = np.zeros(N)
        self.decoding_tree[0].u = np.zeros(N, dtype=int)

        # Label child nodes
        for i in range(self.total_nodes - self.N):
            self.decoding_tree[i].left = self.decoding_tree[2 * self.decoding_tree[i].idx + 1]
            self.decoding_tree[i].right = self.decoding_tree[2 * self.decoding_tree[i].idx + 2]

        # Label parents
        for i in range(1, self.total_nodes):
            parent_idx = (self.decoding_tree[i].idx - 1 - (i % 2 == 0)) // 2
            self.decoding_tree[i].parent = self.decoding_tree[parent_idx]

            # initialize node data
            self.decoding_tree[i].L = np.zeros(len(self.decoding_tree[parent_idx].L) // 2, dtype=int)
            self.decoding_tree[i].u = np.zeros(len(self.decoding_tree[parent_idx].u) // 2, dtype=int)

            # label leaf nodes
            if i >= self.total_nodes - self.N:
                self.decoding_tree[i].leaf_node_idx = i - N + 1

    def set_message(self, msg):
        if len(msg) == self.K:
            # print(self.msg_bits)
            self.message_sent = msg
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
