import numpy as np
from math import *
from QueryPerformanceCounter import *
from polarcodes5G import *

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
        print(myPC)

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

        if np.array_equal(myPC.message_received, myPC.x):
            print("Sent and decoded message matches")
        else:
            raise Exception("Sent and decoded message do not match ")

