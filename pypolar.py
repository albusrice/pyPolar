import numpy as np
from polarcodes import *
from QueryPerformanceCounter import *

# initialise polar code
n = 1024
k = n // 2
myPC = PolarCode(n, k)
myPC.construction_type = 'bb'

# mothercode construction
design_SNR = 4
Construct(myPC, design_SNR)
print(myPC.reliabilities)
print(myPC.frozen_lookup)
print(myPC, "\n\n")

# set message
my_message = np.random.randint(0, 2, k)
myPC.set_message(my_message)
print("The message is:", my_message)

# encode message
performance_counter = QueryPerformanceCounter()

performance_counter.start()
Encode(myPC)
performance_counter.end()
print("The coded message is:", myPC.x)
print("The encoded message is:", myPC.u)

# transmit the codeword
AWGN(myPC, design_SNR)
print("The log-likelihoods are:", myPC.likelihoods)

# decode the received codeword
performance_counter.start()
Decode(myPC)
performance_counter.end()
print("The decoded message is:", myPC.message_received)
print(np.all(np.equal(myPC.message_received, my_message)))

# print(list(myPC.reliabilities))
