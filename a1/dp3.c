
/**
 * @file dp3.c
 * @author Allen Zhang
 * @version 0.1
 * @date 2022-02-11
 */
#include <mkl_cblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
struct timespec start, end;
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

float bdp(long N, float *pA, float *pB)
{
    float R = cblas_sdot(N, pA, 1, pB, 1);
    printf("result %f\n",R);
    return R;
}



int main(int argc, char **argv)
{
    // read arguments and convert to int
    long int vector_size = atoi(argv[1]);
    int measurements_num = atoi(argv[2]);

    // initalize float array
    float pA[vector_size];
    float pB[vector_size];
    for (int i = 0; i < vector_size; i++)
    {
        pA[i] = 1.0;
        pB[i] = 1.0;
    }

    // compute total time used for total number of measures
    double total_time=0.0;
    double half_total_time=0.0;

    // loop by neasurements_num times
    for (int i = 0; i < measurements_num; i++)
    {
        clock_gettime(CLOCK_MONOTONIC, &start);
        bdp(vector_size, pA, pB);
        clock_gettime(CLOCK_MONOTONIC, &end);
        double temp = (((double)end.tv_sec * 1000000000 + (double)end.tv_nsec) - ((double)start.tv_sec * 1000000000 + (double)start.tv_nsec));

        total_time+=temp;
        if (i>=measurements_num/2){
            half_total_time+=temp;
        }
    }
    // compute average execution time

    printf("dp 3\n");

    // total size of pA and pB
    double total_size = 2*vector_size*sizeof(float);
    // <T>
    // double average_exe_time = total_time / measurements_num;
    // millseconds 
    long double second_half_average = half_total_time / (measurements_num/2);

    // compute bandwidth Gb/second
    // size: bytes -> Gb 10^-9; m_second -> second: 10^6 
    double bandwidth=total_size/second_half_average;

    // compute flops: operations/second 
    double flops = ((double)2*vector_size)/second_half_average;

    // N: 1000000 <T>: 9.999999 sec B: 9.999 GB/sec F: 9.999 FLOP/sec
    second_half_average=second_half_average*pow(10,-9);
    printf("N: %ld; <T>: %.9Lf sec; B: %.3f GB/sec; F: %.3f GFlop/sec\n", 
            vector_size, second_half_average,bandwidth,flops);

    return 0;
}
