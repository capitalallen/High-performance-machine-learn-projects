import sys 
import time 
import numpy as np 



if __name__ == "__main__":
    N=int(sys.argv[1])
    measurement_num=int(sys.argv[2])

    # initalize 2 array A and B with size of N 
    A = np.ones(N,dtype=np.float32) 
    B = np.ones(N,dtype=np.float32)

    total_time=[]
    
    # iterate by measurement_num times 
    for i in range(measurement_num):
        # compute execution time 
        start=time.monotonic() 
        np.dot(A,B)
        end=time.monotonic()   
        # add to total_time 
        total_time.append(end-start)
    
    average_time=sum(total_time)/measurement_num
    second_half_average_time=sum(total_time[int(measurement_num/2):])/(int(measurement_num/2))
    bandwidth=(sys.getsizeof(A)+sys.getsizeof(B))/average_time*(10**-9)
    floaps=2*N/average_time
    print("dp 5")
    # N: 1000000 <T>: 9.999999 sec B: 9.999 GB/sec F: 9.999 FLOP/sec
    print("N: {};<T>: {} sec; B: {} GB/sec; F: {} FLOP/sec".format(N,average_time,bandwidth,floaps))
    # print("number of vectors: {}; number of measurements: {}; average execution time: {}".format(N,measurement_num,average_time));
    print("mean for the execution time for the second half of the repetition: {}\n".format(second_half_average_time))
