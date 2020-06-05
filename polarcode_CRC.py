from py5Gpolar import Construct, Encoder, AWGN
import numpy as np
from math import *
from sympy.polys.domains import ZZ
from sympy.polys.galoistools import gf_div, gf_rem
from QueryPerformanceCounter import *


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

N = 2 ** 5
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
# g = lambda a, b, c: b + (1 - 2 * c) * a

Nbiterrs = 0
Nblkerrs = 0
Nblocks = 1

for blk in range(Nblocks):
    myPC = Construct(N, K)
    Q1 = myPC.reliability_sequence
    F = myPC.frozen_bits

    # msg = np.random.randint(0, 2, A)
    msg = np.array([0, 0, 0, 1])
    msg2 = np.zeros(K)
    msg2[:A] = msg
    print(msg)
    quot, rem = gfdeconv(msg2, crcg)
    # print("rem :", rem, len(rem))
    msg2[K-len(rem):] = rem
    msgcrc = msg2
    if len(gfdeconv(msgcrc, crcg)[1]) != 0:
        raise Exception("Encoder CRC FAILED")

    # quot, rem = gfdeconv(np.concatenate([np.zeros(crcL), fliplr(msg)]), crcg)
    # msgcrc = np.concatenate([msg, fliplr(np.concatenate([rem, np.zeros(crcL - len(rem))]))])

    # msgcrc = np.random.randint(0, 2, K)
    myPC.set_message(msgcrc)
    print(myPC)
    # print(gfdeconv(msgcrc, crcg))

    Encoder(myPC)
    print(myPC.u)

    AWGN(myPC, SNR=4)


    # Quantization
    # r = satx(myPC.llrs, rmax)
    # rq = np.round(myPC.llrs / rmax * maxqr)
    #
    rq = np.array([-7.0, -10.0, 6.0, 15.0, -4.0, -8.0, -9.0, 9.0, 10.0, -3.0, 5.0, 6.0, -9.0, 11.0, -9.0, 9.0, -14.0,
                   14.0, 7.0, -11.0, 13.0, 7.0, -18.0, -4.0, 9.0, -8.0, 9.0, -7.0, -2.0, -3.0, 3.0, -10.0]
                  )

    # print(list(rq))

    # nL SC Decoder
    LLR = np.zeros(shape=[nL, n + 1, N], dtype=float)      # beliefs in nL decoders
    ucap = np.zeros(shape=[nL, n + 1, N], dtype=int)       # decisions in nL decoders
    PML = np.infty * np.ones(nL)            # Path metrics
    PML[0] = 0
    ns = np.zeros(2 * N - 1)                # Node state vector

    LLR[:, 0, :] = np.tile(rq, (nL, 1))

    node = 0                                    # start at root
    depth = 0
    done = 0                                    # check if decoder is finished
    performance_counter = QueryPerformanceCounter()
    performance_counter.start()
    while done == 0:
        if depth == n:
            DM = LLR[:, n, node]    # decision metrics
            # print(node, DM[0])
            if node in F:                       # check if node is frozen
                # print("frozen", depth, node)
                ucap[:, n, node] = 0            # set all decisions to 0
                PML += np.abs(DM) * (DM < 0)    # if DM is negative, add |DM|
                print(node, ":", DM, PML,)

            else:
                # print("not frozen", depth, node)
                dec = DM < 0                                    # decisions as per DM
                # print(DM)
                PM2 = np.concatenate([PML, PML + np.abs(DM)])
                # print(PM2)
                PML, pos = mink(PM2, nL)                        # In PM2[:], first nL are as per DM
                                                                # next nL are opposite of DM
                print(node, ":", DM, dec, PML, pos)
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
                # print(a, b)
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
                    # print(Ln.shape)
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
                    # print(ucapl.shape, ucapr, temp * node, (temp * (node + 1)) // 2)
                    ucap[:, depth, temp * node: temp * node + temp // 2] = np.logical_xor(ucapl, ucapr)
                    ucap[:, depth, temp * node + temp // 2: temp * (node + 1)] = ucapr

                    node = node // 2
                    depth -= 1

    performance_counter.end("Decoder")

    # check crc (21.21)
    msg_capl = ucap[:, n, Q1[N - K:]]
    # print(msg_capl)
    cout = 0
    for c1 in range(nL):
        q1, r1 = gfdeconv(msg_capl[c1], crcg)
        # print("r1", r1)
        if len(r1) == 0:
            print(c1)
            msg_cap = msg_capl[cout, :A]
            # print(np.equal(myPC.message_sent[:A], msg_cap))
            print(np.array_equal(myPC.message_sent[:A], msg_cap))
            print("break")
            cout = c1
            break
        if c1 == nL - 1:
            print("CRC failed")




    # msg_cap = ucap[cout, -1, :]

    # print(msg_cap)

