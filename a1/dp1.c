/**
 * @file dp1.c
 * @author Allen Zhang
 * @version 0.1
 * @date 2022-02-11
 */
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
float dp(long int N, float *pA, float *pB)
{
    float R = 0.0;
    int j;
    for (j = 0; j < N; j++)
        R += pA[j] * pB[j];
    return R;
}

// function to return sum of elements
// in an array of size n
double sum(double arr[], int n)
{
    double sum = 0; // initialize sum

    // Iterate through all elements
    // and add them to sum
    for (int i = 0; i < n; i++)
        sum += arr[i];

    return sum;
}

int main(int argc, char **argv)
{
    // read arguments and convert to int
    long int vector_size = atoi(argv[1]);
    int measurements_num = atoi(argv[2]);

    // initalize float array
    float *pA= malloc(sizeof(float) * vector_size);
    float *pB= malloc(sizeof(float) * vector_size);
    for (int i = 0; i < vector_size; i++)
    {
        pA[i] = 1;
        pB[i] = 1;
    }

    // compute total time used for total number of measures
    double total_time_usec[measurements_num];

    // loop by neasurements_num times
    for (int i = 0; i < measurements_num; i++)
    {
        clock_gettime(CLOCK_MONOTONIC, &start);
        dp(vector_size, pA, pB);
        clock_gettime(CLOCK_MONOTONIC, &end);
        double temp = (((double)end.tv_sec * 1000000 + (double)end.tv_nsec / 1000) - ((double)start.tv_sec * 1000000 + (double)start.tv_nsec / 1000));
        double bandwidth_temp = 2*vector_size*sizeof(float)*pow(10,-9)/temp*pow(10,6);

        // printf("index %d; bandwidth: %f, time used: %f\n",i,bandwidth_temp,temp);
        total_time_usec[i] = temp;
    }
    // compute average execution time
    printf("dp 1\n");

    double total_time = sum(total_time_usec, measurements_num);
    double first_half_sum = sum(total_time_usec, measurements_num / 2);
    // total size of pA and pB
    double total_size = 2*vector_size*sizeof(float)*pow(10,-9);
    // <T>
    // double average_exe_time = total_time / measurements_num;
    // millseconds 
    double second_half_average = (total_time - first_half_sum) / (measurements_num/2);

    // compute bandwidth Gb/second
    // size: bytes -> Gb 10^-9; m_second -> second: 10^6 

    double bandwidth=total_size/second_half_average*pow(10,6);

    // compute flops: operations/second 
    double flops = ((double)2*vector_size)/second_half_average*pow(10,6);

    // N: 1000000 <T>: 9.999999 sec B: 9.999 GB/sec F: 9.999 FLOP/sec
    printf("N: %ld; <T>: %f sec; B: %.3f GB/sec; F: %.3f GFlop/sec\n", 
            vector_size, second_half_average,bandwidth,flops*pow(10,-9));

    return 0;
}