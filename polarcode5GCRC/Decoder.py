import numpy as np
from polarcode5GCRC.utils import mink, gfdeconv


def satx(x, th):
    return np.minimum(np.maximum(x, -th), th)


def f(a, b):
    # min sum function
    return (1 - 2 * (a < 0)) * (1 - 2 * (b < 0)) * np.minimum(np.abs(a), np.abs(b))


class Decoder:
    def __init__(self, myPC, list_decoder=False, rmax=4, maxqr=31):

        self.list_decoder = list_decoder
        self.myPC = myPC
        self.maxqr = maxqr

        # Quantization
        rq = self.myPC.llrs

        # r = satx(self.myPC.llrs, rmax)
        # rq = np.round(r / rmax * self.maxqr)
        # rq = np.array([-7.0, -10.0, 6.0, 15.0, -4.0, -8.0, -9.0, 9.0, 10.0, -3.0, 5.0, 6.0, -9.0, 11.0, -9.0, 9.0, -14.0,
        #                14.0, 7.0, -11.0, 13.0, 7.0, -18.0, -4.0, 9.0, -8.0, 9.0, -7.0, -2.0, -3.0, 3.0, -10.0])

        self.myPC.LLR[:, 0, :] = np.tile(rq, (self.myPC.Ln, 1))

        self.polar_decoder()

        if self.list_decoder:
            # Extract message
            K = self.myPC.K + self.myPC.crcl
            Q1 = self.myPC.reliability_sequence
            msg_capl = self.myPC.hat_u[:self.myPC.Ln, self.myPC.depth, Q1[self.myPC.N - K:]]

            cout = 0

            for c1 in range(self.myPC.Ln):
                # check if messages passes CRC check
                q1, r1 = gfdeconv(msg_capl[c1], self.myPC.crc_polynomial)
                if len(r1) == 0:
                    cout = c1
                    # print("choice : ", cout)
                    break
                if c1 == self.myPC.Ln - 1:
                    print("CRC failed")

            self.myPC.message_received = self.myPC.hat_u[cout, -1, :]

        else:
            self.myPC.message_received = self.myPC.hat_u[0, -1, :]

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
                    # print(curr.idx, curr.depth, curr.length)
                    _next = curr.left
                    llr = self.myPC.LLR[:, curr.depth, curr.start: curr.end]    # incoming beliefs
                    a = llr[:, :curr.length // 2]
                    b = llr[:, curr.length // 2:]
                    self.myPC.LLR[:, _next.depth, _next.start:_next.end] = f(a, b)
                    # print(self.myPC.LLR[0])

                    # print("traverse left:", a, b)

                else:
                    # leaf node
                    dec_mat = self.myPC.LLR[:, curr.depth, curr.start]
                    if self.list_decoder:
                        # Successive list cancellation
                        if curr.start in self.myPC.frozen_bits:
                            # set all decisions to 0
                            self.myPC.hat_u[:, curr.depth, curr.start] = 0

                            # if DM is negative, add |DM| penalty
                            self.myPC.PML[:self.myPC.Ln] += np.abs(dec_mat) * (dec_mat < 0)

                            # print(curr.start, ":", dec_mat, self.myPC.PML[:self.myPC.Ln])

                        else:
                            # Make decisions at leaf nodes
                            dec = dec_mat < 0
                            self.myPC.PML[self.myPC.Ln:] = self.myPC.PML[:self.myPC.Ln] + np.abs(dec_mat)
                            self.myPC.PML[:self.myPC.Ln], pos = mink(self.myPC.PML, self.myPC.Ln)

                            # print(curr.start, ":", dec_mat, dec, self.myPC.PML[:self.myPC.Ln], pos)

                            pos1 = pos >= self.myPC.Ln              # surviving with opposite of DM: 1, if pos > nL
                            pos[pos1] = pos[pos1] - self.myPC.Ln    # adjust index
                            dec = dec[pos]                          # decision of survivors
                            dec[pos1] = 1 - dec[pos1]               # flip decision for opposite DM

                            # rearrange the decoder states
                            self.myPC.LLR = self.myPC.LLR[pos, :, :]
                            self.myPC.hat_u = self.myPC.hat_u[pos, :, :]
                            self.myPC.hat_u[:, curr.depth, curr.start] = dec

                    else:
                        # successive list cancellation
                        if curr.start in self.myPC.frozen_bits:
                            self.myPC.hat_u[:, curr.depth, curr.start] = 0
                        else:
                            self.myPC.hat_u[:, curr.depth, curr.start] = dec_mat < 0

                    _next = curr.parent

            elif curr.left == prev:
                # traverse right
                _next = curr.right
                llr = self.myPC.LLR[:, curr.depth, curr.start: curr.end]    # incoming beliefs
                # print(llr)

                a = llr[:, :curr.length // 2]
                b = llr[:, curr.length // 2:]
                hat_u = self.myPC.hat_u[:, curr.depth + 1, curr.left.start:curr.left.end]
                # print("traverse right :", a, b, hat_u)

                self.myPC.LLR[:, _next.depth, _next.start:_next.end] = self.g(a, b, hat_u)
            else:
                # root node
                # compute u
                lhat_u = self.myPC.hat_u[:, curr.depth + 1, curr.left.start: curr.left.end]             # left child u
                rhat_u = self.myPC.hat_u[:, curr.depth + 1, curr.right.start: curr.right.end]           # right child u
                # print("compute u :", lhat_u, rhat_u)

                # current node u
                to_idx = curr.start + curr.length // 2
                self.myPC.hat_u[:, curr.depth, curr.start: to_idx] = np.logical_xor(lhat_u, rhat_u)
                self.myPC.hat_u[:, curr.depth, to_idx : curr.end] = rhat_u

                _next = curr.parent

            prev = curr
            curr = _next

    def g(self, a, b, c):
        if self.list_decoder:
            return satx(b + (1 - 2 * c) * a, self.maxqr)
        else:
            return b + (1 - 2 * c) * a
