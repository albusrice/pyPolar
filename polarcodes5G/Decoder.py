import numpy as np


class Decoder:
    def __init__(self, myPC):
        self.myPC = myPC
        self.myPC.decoding_tree[0].L = self.myPC.llrs

        self.polar_decoder()

        self.myPC.message_received = np.array([
            t.u[0] for t in self.myPC.decoding_tree[self.myPC.total_nodes - self.myPC.N:]
        ])

    def polar_decoder(self):
        """
        Successive cancellation decoder using binary tree structure
        :return:
        """
        if not self.myPC.decoding_tree[0]:
            return

        prev = None
        curr = self.myPC.decoding_tree[0]
        _next = None
        while curr:
            if not prev or prev.left == curr or prev.right == curr:
                # traverse left
                if curr.left:
                    _next = curr.left
                    _next.L = self.f(curr.L[:len(curr.L) // 2], curr.L[len(curr.L) // 2:])
                else:
                    # leaf node
                    if curr.leaf_node_idx in self.myPC.frozen_bits:
                        curr.u[0] = 0
                    else:
                        curr.u[0] = curr.L[0] < 0
                    _next = curr.parent

            elif curr.left == prev:
                # traverse right
                _next = curr.right
                _next.L = self.g(curr.L[:len(curr.L) // 2], curr.L[len(curr.L) // 2:], curr.left.u)
            else:
                # root node
                # compute u
                curr.u[:len(curr.u) // 2] = np.logical_xor(curr.left.u, curr.right.u)
                curr.u[len(curr.u) // 2:] = curr.right.u
                _next = curr.parent

            prev = curr
            curr = _next

    def f(self, a, b):
        # min sum function
        return (1 - 2 * (a < 0)) * (1 - 2 * (b < 0)) * np.minimum(np.abs(a), np.abs(b))

    def g(self, a, b, c):
        return b + (1 - 2 * c) * a
