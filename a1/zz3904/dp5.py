import sys 
import time 
import numpy as np 



if __name__ == "__main__":
    N=int(sys.argv[1])
    measurement_num=int(sys.argv[2])

    # initalize 2 array A and B with size of N 
    A = np.ones(N,dtype=np.float32) 
    B = np.ones(N,dtype=np.float32)

    total_time=0.0
    half_total=0.0
    
    # iterate by measurement_num times 
    for i in range(measurement_num):
        # compute execution time 
        start=time.monotonic_ns() 
        np.dot(A,B)
        end=time.monotonic_ns() 
 
        # add to total_time 
        total_time+=end-start
        if i>=measurement_num/2:
            half_total+=end-start
    
    
    second_half_average_time=half_total/(measurement_num/2)

    bandwidth=(sys.getsizeof(A)+sys.getsizeof(B))/second_half_average_time
    flops=2*N/second_half_average_time
    print("dp 5")
    # N: 1000000 <T>: 9.999999 sec B: 9.999 GB/sec F: 9.999 FLOP/sec
    print("N: {};<T>: {} sec; B: {} GB/sec; F: {} GFLOP/sec".format(N,second_half_average_time*pow(10,-9),bandwidth,flops))    # print("number of vectors: {}; number of measurements: {}; average execution time: {}".format(N,measurement_num,average_time));

