import numpy as np
from math import *
from QueryPerformanceCounter import *
from polarcode5GCRC import *

if __name__ == '__main__':
    for _ in range(1):
        snr = 4
        n = 2 ** 10
        k = n // 2 - 12

        performance_counter = QueryPerformanceCounter()

        # construct polar code structure
        myPC = Construct(n, k, crc=True, num_decoders=4)

        # Generate random message
        my_msg = np.random.randint(0, 2, k)

        myPC.set_message(my_msg)
        print(myPC)

        # encode message
        performance_counter.start()
        Encoder(myPC)
        performance_counter.end("Encoder")

        # add AWGN noise
        AWGN(myPC, snr)

        # print(list(myPC.llrs))

        performance_counter.start()
        Decoder(myPC, list_decoder=False)

        performance_counter.end("Decoder")

        # print("recieved message: ", list(myPC.message_received))
        # print(np.equal(myPC.message_received, myPC.x))

        if np.array_equal(myPC.message_received, myPC.x):
            print("Sent and decoded message matches")
        else:
            raise Exception("Sent and decoded message do not match ")


