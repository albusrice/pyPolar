from polarcodes5G import Construct, Encoder, AWGN
import numpy as np
from math import *
from sympy.polys.domains import ZZ
from sympy.polys.galoistools import gf_div
from QueryPerformanceCounter import *


performance_counter = QueryPerformanceCounter()

def gfdeconv(a, b):
    return gf_div(ZZ.map(a), ZZ.map(b), 2, ZZ)


def fliplr(a):
    return a[::-1]


def mink(arr, k):
    idx = arr.argsort()[:k]
    return arr[idx], idx


# N = 2 ** 6
# print("N =", N)
# K = N // 2
# Rate = K / N

N = 2 ** 10
crcL = 11
A = N // 2 - 1 - crcL
crcg = np.array([1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1])   # x ** 11 + x ** 10 + x ** 9 + x ** 5 + 1
Rate = A / N
K = A + crcL

n = int(log2(N))
EbNodB = 0.1
rmax = 4
maxqr = 31
nL = 4


EbNo = 10.0 ** (EbNodB / 10.0)
sigma = sqrt(1 / (2 * Rate * EbNo))

satx = lambda x, th : np.minimum(np.maximum(x, -th), th)
f = lambda a, b : (1 - 2 * (a < 0)) * (1 - 2 * (b < 0)) * np.minimum(np.abs(a), np.abs(b))
g = lambda a, b, c : satx(b + (1 - 2 * c) * a, maxqr)

Nbiterrs = 0
Nblkerrs = 0
Nblocks = 1

for blk in range(Nblocks):
    myPC = Construct(N, K)
    Q1 = myPC.reliability_sequence
    F = myPC.frozen_bits

    # Generate random message
    msg = np.random.randint(0, 2, A)
    msg2 = np.zeros(K)
    msg2[:A] = msg

    # add CRC to message
    quot, rem = gfdeconv(msg2, crcg)
    msg2[K-len(rem):] = rem
    msgcrc = msg2
    if len(gfdeconv(msgcrc, crcg)[1]) != 0:
        raise Exception("Encoder CRC FAILED")

    myPC.set_message(msgcrc)
    print(myPC)
    # print(gfdeconv(msgcrc, crcg))
    performance_counter.start()
    Encoder(myPC)
    performance_counter.end("Encoder")
    # print(myPC.u)

    AWGN(myPC, SNR=4)

    # nL SC Decoder
    LLR = np.zeros(shape=[nL, n + 1, N], dtype=float)      # beliefs in nL decoders
    ucap = np.zeros(shape=[nL, n + 1, N], dtype=int)       # decisions in nL decoders
    PML = np.infty * np.ones(nL)            # Path metrics
    PML[0] = 0
    ns = np.zeros(2 * N - 1)                # Node state vector

    performance_counter.start()

    # myPC.llrs = np.array([0.7850752429487079, -0.21151714052461235, 2.616214794701895, 0.6576594738869361,
    #                        -0.39216604607850236, -0.18671349604561738, 1.3064589492280385, 2.250322679378182,
    #                        0.31898528852890307, 0.6915918975469443, -1.282812087702204, 2.9970362693358634,
    #                        -0.08034465273633495, -1.8623270309319788, -1.0826029272054507, 1.4262679407189522,
    #                        2.6942254294821764, 0.6803684430042729, 1.1255818970236242, -5.380262380680362,
    #                        0.036267527436991354, -1.0408785492161312, 1.7158288740098628, 3.2726947987128434,
    #                        0.5764644459941721, 0.6130516054151076, -0.5339639176947478, 0.5876979570085821,
    #                        -0.34522778936242116, -2.5901042666509717, 0.9882757794628532, 1.4184460184293997
    #                       ])
    # Quantization
    r = satx(myPC.llrs, rmax)
    rq = np.round(r / rmax * maxqr)

    # rq = np.array([-7.0, -10.0, 6.0, 15.0, -4.0, -8.0, -9.0, 9.0, 10.0, -3.0, 5.0, 6.0, -9.0, 11.0, -9.0, 9.0, -14.0,
    #                14.0, 7.0, -11.0, 13.0, 7.0, -18.0, -4.0, 9.0, -8.0, 9.0, -7.0, -2.0, -3.0, 3.0, -10.0])
    # print(list(rq))

    LLR[:, 0, :] = np.tile(rq, (nL, 1))

    node = 0                                    # start at root
    depth = 0
    done = 0                                    # check if decoder is finished

    while done == 0:
        if depth == n:
            DM = LLR[:, n, node]    # decision metrics
            # print(node, DM[0])
            if node in F:                       # check if node is frozen
                # print("frozen", depth, node)
                ucap[:, n, node] = 0            # set all decisions to 0
                PML += np.abs(DM) * (DM < 0)    # if DM is negative, add |DM|
                # print(node, ":", DM, PML,)

            else:
                # print("not frozen", depth, node)
                dec = DM < 0                                    # decisions as per DM
                # print(DM)
                PM2 = np.concatenate([PML, PML + np.abs(DM)])
                # print(PM2)
                PML, pos = mink(PM2, nL)                        # In PM2[:], first nL are as per DM
                                                                # next nL are opposite of DM
                # print(node, ":", DM, dec, PML, pos)
                pos1 = pos >= nL                                # surviving with opposite of DM: 1, if pos is above nL
                pos[pos1] = pos[pos1] - nL                      # adjust index
                dec = dec[pos]                                  # decision of survivors
                dec[pos1] = 1 - dec[pos1]                       # flip decision for opposite DM
                LLR = LLR[pos, :, :]                            # rearrange the decoder states
                # print(pos)
                ucap = ucap[pos, :, :]
                # print(dec, PM2)
                ucap[:, n, node] = dec

            if node == N - 1:
                done = 1
            else:
                node = floor(node / 2)
                depth -= 1

        else:
            # non-leaf (8.54)
            npos = (2 ** depth - 1) + node + 1                  # position of node in node state vector
            if ns[npos] == 0:                                   # step L and go to left child
                temp = 2 ** (n - depth)
                # print("L", depth, node, temp, temp * node, temp * (node + 1))
                Ln = LLR[:, depth, temp * node: temp * (node + 1)]      # incoming beliefs
                # print(Ln.shape, temp)
                a = Ln[:, :temp // 2]                           # split beliefs into 2
                b = Ln[:, temp // 2:]
                # print("traverse left:", a, b)
                node *= 2
                depth += 1
                temp = temp // 2
                # print(a.shape, b.shape)
                # print(np.shape(LLR[:, depth, temp * node: temp * (node + 1)]), temp * node, temp * (node + 1))
                LLR[:, depth, temp * node: temp * (node + 1)] = f(a, b)
                ns[npos] = 1
            else:
                if ns[npos] == 1:                              # step R and go to the right child
                    temp = 2 ** (n - depth)
                    # print("R", depth, node, temp, temp * node, temp * (node + 1))
                    Ln = LLR[:, depth, temp * node: temp * (node + 1)]
                    # Ln = np.squeeze(LLR[:, depth, temp * node: temp * (node + 1)])  # incoming beliefs
                    # print(Ln)
                    # print(Ln.shape, temp)
                    a = Ln[:, :temp // 2]
                    b = Ln[:, temp // 2:]
                    lnode = 2 * node                            # left child
                    ldepth = depth + 1
                    ltemp = temp // 2                           # incoming belief length for right child
                    ucapn = ucap[:, ldepth, ltemp * lnode: ltemp * (lnode + 1)]
                    node = node * 2 + 1                         # next node: right child
                    depth += 1
                    temp = temp // 2
                    # print(a.shape, b.shape, ucapn.shape)
                    # print(np.shape(LLR[:, depth, temp * node: temp * (node + 1)]),  temp * node, temp * (node + 1))
                    # print(np.shape(g(a, b, ucapn)))
                    # print("traverse right :", a, b, ucapn)
                    # self.myPC.l
                    LLR[:, depth, temp * node: temp * (node + 1)] = g(a, b, ucapn)  # g and storage

                    ns[npos] = 2

                else:                                           # step U and go to parent
                    # print("U", depth, node)
                    temp = 2 ** (n - depth)
                    lnode = 2 * node                            # left child
                    rnode = 2 * node + 1                        # right child
                    cdepth = depth + 1
                    ctemp = temp // 2
                    ucapl = ucap[:, cdepth, ctemp * lnode: ctemp * (lnode + 1)]
                    ucapr = ucap[:, cdepth, ctemp * rnode: ctemp * (rnode + 1)]

                    # print("compute u :", ucapl, ucapr)
                    # print(ucapl.shape, ucapr, temp * node, (temp * (node + 1)) // 2)
                    ucap[:, depth, temp * node: temp * node + temp // 2] = np.logical_xor(ucapl, ucapr)
                    ucap[:, depth, temp * node + temp // 2: temp * (node + 1)] = ucapr

                    node = node // 2
                    depth -= 1

    # check crc (21.21)
    msg_capl = ucap[:, n, Q1[N - K:]]
    # print(msg_capl)
    cout = 0
    for c1 in range(nL):
        q1, r1 = gfdeconv(msg_capl[c1], crcg)
        # print("r1", r1)
        if len(r1) == 0:
            msg_cap = msg_capl[cout, :A]
            # print(np.equal(myPC.message_sent[:A], msg_cap))


            cout = c1
            break
        if c1 == nL - 1:
            print("CRC failed")

    performance_counter.end("Decoder")

    print(cout)
    # print(msg_capl[cout, :A])
    print(np.array_equal(myPC.message_sent[:A], msg_capl[cout, :A]))


    # msg_cap = ucap[cout, -1, :]

    # print(msg_cap)

