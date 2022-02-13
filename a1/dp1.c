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
float dp(long N, float *pA, float *pB)
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
    double total_time_usec[5000]; 

    // loop by neasurements_num times 
    for (int i=0;i<measurements_num;i++){
        clock_gettime(CLOCK_MONOTONIC,&start); 
        dp(vector_size, pA, pB);
        clock_gettime(CLOCK_MONOTONIC,&end);
        double temp = (((double)end.tv_sec * 1000000 + (double)end.tv_nsec / 1000) - ((double)start.tv_sec * 1000000 + (double)start.tv_nsec / 1000));
        // printf("index %d; time used: %f\n",i,temp);
        total_time_usec[i]=temp;
    }
    // compute average execution time 
    printf("dp 1\n");

    double total_time=sum(total_time_usec,measurements_num);
    double first_half_sum=sum(total_time_usec,measurements_num/2);

    double average_exe_time = total_time/measurements_num;
    double second_half_average=(total_time-first_half_sum)/2;

    printf("number of vectors: %d; number of measurements: %d; average execution time: %f\n",vector_size,measurements_num,average_exe_time);
    printf("neam for the execution time for the second half of the repetition: %f\n",second_half_average);
    
    return 0;
}