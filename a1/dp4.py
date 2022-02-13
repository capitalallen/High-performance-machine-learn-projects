import sys 
import time 
import numpy as np 


def dp(N,A,B): 
    R = 0.0
    for j in range(0,N):
        R += A[j]*B[j]
    return R

if __name__ == "__main__":
    N=int(sys.argv[1])
    measurement_num=int(sys.argv[2])

    # initalize 2 array A and B with size of N 
    A = np.ones(N,dtype=np.float32) 
    B = np.ones(N,dtype=np.float32)

    total_time=0.0
    
    # iterate by measurement_num times 
    for i in range(measurement_num):
        # compute execution time 
        start=time.monotonic() 
        dp(N,A,B)
        end=time.monotonic()   
        # add to total_time 
        total_time+=end-start 
    
    average_time=total_time/measurement_num
    print("dp 4")
    print("number of vectors: {}; number of measurements: {}; average execution time: {}".format(N,measurement_num,average_time));
