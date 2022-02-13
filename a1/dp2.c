/**
 * @file dp1.c
 * @author Allen Zhang
 * @version 0.1
 * @date 2022-02-11
 */
#include <stdio.h>
#include <stdlib.h>
struct timespec start, end;
#include <time.h>
/**
 * @brief micro-benchmark that measures dot-product performance
 *
 * inputs:
 *  N - length of loop (long)
 *  pA - array of A (pointer)
 *  pB - array of B (pointer)
 * output:
 *  R - dot product of array A and B
 */

float dpunroll(long N, float *pA, float *pB) { 
    float R = 0.0;
    int j;
    for (j=0;j<N;j+=4)
    R += pA[j]*pB[j] + pA[j+1]*pB[j+1] + pA[j+2]*pB[j+2] + pA[j+3] * pB[j+3]; 
    return R;
}

int main(int argc, char **argv)
{
    // read arguments and convert to int
    int vector_size = atoi(argv[1]);
    int measurements_num = atoi(argv[2]);

    // initalize float array
    float pA[vector_size];
    float pB[vector_size];
    for (int i = 0; i < vector_size; i++)
    {
        pA[i] = 1;
        pB[i] = 1;
    }

    // compute total time used for total number of measures 
    double total_time_usec = 0; 

    // loop by neasurements_num times 
    for (int i=0;i<measurements_num;i++){
        clock_gettime(CLOCK_MONOTONIC,&start); 
        dpunroll(vector_size, pA, pB);
        clock_gettime(CLOCK_MONOTONIC,&end);
        double temp = (((double)end.tv_sec * 1000000 + (double)end.tv_nsec / 1000) - ((double)start.tv_sec * 1000000 + (double)start.tv_nsec / 1000));
        // printf("index %d; time used: %f\n",i,temp);
        total_time_usec+=temp;
    }
    // compute average execution time 
    double average_exe_time = total_time_usec/measurements_num;
    printf("dp 2\n");
    printf("number of vectors: %d; number of measurements: %d; average execution time: %f\n",vector_size,measurements_num,average_exe_time);
    return 0;
}