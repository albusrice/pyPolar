import numpy as np
from math import *
from QueryPerformanceCounter import *


def format_list(a, newline_idx=8):
    string = ''
    count = 0
    for i in range(len(a)):
        string += " {:>4}, ".format(a[i])
        count += 1
        if count % newline_idx == 0 or i == len(a) - 1:
            string += "\n"

    return string + "\n"


class BTreeNode:
    def __init__(self, idx):
        self.idx = idx
        self.parent = None
        self.left = None
        self.right = None
        self.leaf_node_idx = None
        self.L = None
        self.u = None

    def __str__(self):
        return str(self.idx)


class Construct:
    def __init__(self, N, K):
        self.reliability_sequence_full = np.array([
            0,     1,     2,     4,     8,    16,    32,     3,
            5,    64,     9,     6,    17,    10,    18,   128,
            12,    33,    65,    20,   256,    34,    24,    36,
            7,   129,    66,   512,    11,    40,    68,   130,
            19,    13,    48,    14,    72,   257,    21,   132,
            35,   258,    26,   513,    80,    37,    25,    22,
            136,   260,   264,    38,   514,    96,    67,    41,
            144,    28,    69,    42,   516,    49,    74,   272,
            160,   520,   288,   528,   192,   544,    70,    44,
            131,    81,    50,    73,    15,   320,   133,    52,
            23,   134,   384,    76,   137,    82,    56,    27,
            97,    39,   259,    84,   138,   145,   261,    29,
            43,    98,   515,    88,   140,    30,   146,    71,
            262,   265,   161,   576,    45,   100,   640,    51,
            148,    46,    75,   266,   273,   517,   104,   162,
            53,   193,   152,    77,   164,   768,   268,   274,
            518,    54,    83,    57,   521,   112,   135,    78,
            289,   194,    85,   276,   522,    58,   168,   139,
            99,    86,    60,   280,    89,   290,   529,   524,
            196,   141,   101,   147,   176,   142,   530,   321,
            31,   200,    90,   545,   292,   322,   532,   263,
            149,   102,   105,   304,   296,   163,    92,    47,
            267,   385,   546,   324,   208,   386,   150,   153,
            165,   106,    55,   328,   536,   577,   548,   113,
            154,    79,   269,   108,   578,   224,   166,   519,
            552,   195,   270,   641,   523,   275,   580,   291,
            59,   169,   560,   114,   277,   156,    87,   197,
            116,   170,    61,   531,   525,   642,   281,   278,
            526,   177,   293,   388,    91,   584,   769,   198,
            172,   120,   201,   336,    62,   282,   143,   103,
            178,   294,    93,   644,   202,   592,   323,   392,
            297,   770,   107,   180,   151,   209,   284,   648,
            94,   204,   298,   400,   608,   352,   325,   533,
            155,   210,   305,   547,   300,   109,   184,   534,
            537,   115,   167,   225,   326,   306,   772,   157,
            656,   329,   110,   117,   212,   171,   776,   330,
            226,   549,   538,   387,   308,   216,   416,   271,
            279,   158,   337,   550,   672,   118,   332,   579,
            540,   389,   173,   121,   553,   199,   784,   179,
            228,   338,   312,   704,   390,   174,   554,   581,
            393,   283,   122,   448,   353,   561,   203,    63,
            340,   394,   527,   582,   556,   181,   295,   285,
            232,   124,   205,   182,   643,   562,   286,   585,
            299,   354,   211,   401,   185,   396,   344,   586,
            645,   593,   535,   240,   206,    95,   327,   564,
            800,   402,   356,   307,   301,   417,   213,   568,
            832,   588,   186,   646,   404,   227,   896,   594,
            418,   302,   649,   771,   360,   539,   111,   331,
            214,   309,   188,   449,   217,   408,   609,   596,
            551,   650,   229,   159,   420,   310,   541,   773,
            610,   657,   333,   119,   600,   339,   218,   368,
            652,   230,   391,   313,   450,   542,   334,   233,
            555,   774,   175,   123,   658,   612,   341,   777,
            220,   314,   424,   395,   673,   583,   355,   287,
            183,   234,   125,   557,   660,   616,   342,   316,
            241,   778,   563,   345,   452,   397,   403,   207,
            674,   558,   785,   432,   357,   187,   236,   664,
            624,   587,   780,   705,   126,   242,   565,   398,
            346,   456,   358,   405,   303,   569,   244,   595,
            189,   566,   676,   361,   706,   589,   215,   786,
            647,   348,   419,   406,   464,   680,   801,   362,
            590,   409,   570,   788,   597,   572,   219,   311,
            708,   598,   601,   651,   421,   792,   802,   611,
            602,   410,   231,   688,   653,   248,   369,   190,
            364,   654,   659,   335,   480,   315,   221,   370,
            613,   422,   425,   451,   614,   543,   235,   412,
            343,   372,   775,   317,   222,   426,   453,   237,
            559,   833,   804,   712,   834,   661,   808,   779,
            617,   604,   433,   720,   816,   836,   347,   897,
            243,   662,   454,   318,   675,   618,   898,   781,
            376,   428,   665,   736,   567,   840,   625,   238,
            359,   457,   399,   787,   591,   678,   434,   677,
            349,   245,   458,   666,   620,   363,   127,   191,
            782,   407,   436,   626,   571,   465,   681,   246,
            707,   350,   599,   668,   790,   460,   249,   682,
            573,   411,   803,   789,   709,   365,   440,   628,
            689,   374,   423,   466,   793,   250,   371,   481,
            574,   413,   603,   366,   468,   655,   900,   805,
            615,   684,   710,   429,   794,   252,   373,   605,
            848,   690,   713,   632,   482,   806,   427,   904,
            414,   223,   663,   692,   835,   619,   472,   455,
            796,   809,   714,   721,   837,   716,   864,   810,
            606,   912,   722,   696,   377,   435,   817,   319,
            621,   812,   484,   430,   838,   667,   488,   239,
            378,   459,   622,   627,   437,   380,   818,   461,
            496,   669,   679,   724,   841,   629,   351,   467,
            438,   737,   251,   462,   442,   441,   469,   247,
            683,   842,   738,   899,   670,   783,   849,   820,
            728,   928,   791,   367,   901,   630,   685,   844,
            633,   711,   253,   691,   824,   902,   686,   740,
            850,   375,   444,   470,   483,   415,   485,   905,
            795,   473,   634,   744,   852,   960,   865,   693,
            797,   906,   715,   807,   474,   636,   694,   254,
            717,   575,   913,   798,   811,   379,   697,   431,
            607,   489,   866,   723,   486,   908,   718,   813,
            476,   856,   839,   725,   698,   914,   752,   868,
            819,   814,   439,   929,   490,   623,   671,   739,
            916,   463,   843,   381,   497,   930,   821,   726,
            961,   872,   492,   631,   729,   700,   443,   741,
            845,   920,   382,   822,   851,   730,   498,   880,
            742,   445,   471,   635,   932,   687,   903,   825,
            500,   846,   745,   826,   732,   446,   962,   936,
            475,   853,   867,   637,   907,   487,   695,   746,
            828,   753,   854,   857,   504,   799,   255,   964,
            909,   719,   477,   915,   638,   748,   944,   869,
            491,   699,   754,   858,   478,   968,   383,   910,
            815,   976,   870,   917,   727,   493,   873,   701,
            931,   756,   860,   499,   731,   823,   922,   874,
            918,   502,   933,   743,   760,   881,   494,   702,
            921,   501,   876,   847,   992,   447,   733,   827,
            934,   882,   937,   963,   747,   505,   855,   924,
            734,   829,   965,   938,   884,   506,   749,   945,
            966,   755,   859,   940,   830,   911,   871,   639,
            888,   479,   946,   750,   969,   508,   861,   757,
            970,   919,   875,   862,   758,   948,   977,   923,
            972,   761,   877,   952,   495,   703,   935,   978,
            883,   762,   503,   925,   878,   735,   993,   885,
            939,   994,   980,   926,   764,   941,   967,   886,
            831,   947,   507,   889,   984,   751,   942,   996,
            971,   890,   509,   949,   973,  1000,   892,   950,
            863,   759,  1008,   510,   979,   953,   763,   974,
            954,   879,   981,   982,   927,   995,   765,   956,
            887,   985,   997,   986,   943,   891,   998,   766,
            511,   988,  1001,   951,  1002,   893,   975,   894,
            1009,   955,  1004,  1010,   957,   983,   958,   987,
            1012,   999,  1016,   767,   989,  1003,   990,  1005,
            959,  1011,  1013,   895,  1006,  1014,  1017,  1018,
            991,  1020,  1007,  1015,  1019,  1021,  1022,  1023,
        ], dtype=int)

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


class AWGN:
    def __init__(self, myPC, SNR):
        self.myPC = myPC
        self.EbNodB = SNR

        self.transmit()

    def transmit(self):
        rate = self.myPC.K / self.myPC.N
        EbNo = 10.0 ** (self.EbNodB / 10.0)
        sigma = sqrt(1.0 / (2.0 * rate * EbNo))

        self.myPC.llrs = 1 - 2.0 * self.myPC.u                      # BPSK bit symbol conversion
        self.myPC.llrs += sigma * np.random.randn(self.myPC.N)      # Add AWGN channel noise


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
                    performance_counter.start()
                    _next = curr.left
                    _next.L = self.f(curr.L[:len(curr.L) // 2], curr.L[len(curr.L) // 2:])
                    performance_counter.end("left")
                else:
                    performance_counter.start()
                    # leaf node
                    if curr.leaf_node_idx in self.myPC.frozen_bits:
                        curr.u[0] = 0
                    else:
                        curr.u[0] = curr.L[0] < 0
                    _next = curr.parent
                    performance_counter.end("leaf")

            elif curr.left == prev:
                # traverse right
                performance_counter.start()
                _next = curr.right
                _next.L = self.g(curr.L[:len(curr.L) // 2], curr.L[len(curr.L) // 2:], curr.left.u)
                performance_counter.end("right")

            else:
                # root node
                # compute u
                performance_counter.start()
                curr.u[:len(curr.u) // 2] = np.logical_xor(curr.left.u, curr.right.u)
                curr.u[len(curr.u) // 2:] = curr.right.u
                _next = curr.parent
                performance_counter.end("u")

            prev = curr
            curr = _next

    def f(self, a, b):
        # min sum function
        return (1 - 2 * (a < 0)) * (1 - 2 * (b < 0)) * np.minimum(np.abs(a), np.abs(b))

    def g(self, a, b, c):
        return b + (1 - 2 * c) * a


if __name__ == '__main__':
    for _ in range(1):
        snr = 4
        n = 16
        k = n // 2
        performance_counter = QueryPerformanceCounter()
        myPC = Construct(n, k)
        # my_msg = [1, 1, 0, 1, 1, 0, 0, 1]
        my_msg = np.random.randint(0, 2, k)

        myPC.set_message(my_msg)
        # print(myPC)

        performance_counter.start()
        Encoder(myPC)
        performance_counter.end("Encoder")
        # print('u:', myPC.u)

        AWGN(myPC, snr)
        # myPC.llrs = np.array([-0.3449045047482715, -1.5038744272566131, -1.4684420902979543, -0.499699587344099, -1.3661878661461808, 1.7961680342252324, -0.350996604515565, 0.8418644542913243, 0.11279407884180914, 1.2084472436254747, -1.0970962154668251, 0.847794235482955, -0.6891956421072841, -1.5977724052059754, -1.5341548707747088, -1.5428690071144406])
        # print(list(myPC.llrs))
        performance_counter.start()
        Decoder(myPC)
        performance_counter.end("Decoder")
        # print("recieved message: ", myPC.message_received)
        print(np.array_equal(myPC.message_received, myPC.x))

