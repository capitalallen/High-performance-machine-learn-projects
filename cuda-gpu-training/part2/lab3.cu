#include <stdio.h>
#include <time.h>
#include <assert.h>


#define H 1024
#define W 1024
#define C 3
#define FH 3
#define FW 3
#define K 64
#define TH 32
#define TW 32

/**
 * @brief create image in the memory
 *
 * @param h
 * @param w
 * @param c
 * @param it
 */
void load_image(int h, int w, int c, double *it)
{
	for (int ki = 0; ki < c; ++ki)
	{
		for (int j = 0; j < h; ++j)
		{
			for (int i = 0; i < w; ++i)
			{
				it[ki * w * h + j * w + i] = ki * (i + j);
			}
		}
	}
}

static double convert_to_sec(struct timespec *timSec)
{
	double tinSec = (double)timSec->tv_sec;
	double tinNSec = (double)timSec->tv_nsec;
	return tinSec + tinNSec / 1000000000.0;
}

//  implement a convolution of an image using a set of filters.
__global__ void convolution2DKernel(int c, int k, int h, int w, int fh, int fw, double *itg, double *gpuf, double *otg)
{
	int padded_tile_size = (blockDim.x + fw - 1) * (blockDim.y + fh - 1);
	int index = blockDim.x * threadIdx.y + threadIdx.x;
	extern __shared__ double p_img_x[];

	for (int p = index; p < padded_tile_size; p += (blockDim.x) * (blockDim.y))
	{
		int x = (p % (blockDim.x + fw - 1)) + (blockIdx.x * blockDim.x) - ((fw) / 2);
		int y = (p / (blockDim.x + fw - 1)) + (blockIdx.y * blockDim.y) - ((fh) / 2);
		if (x < 0 || x >= w || y < 0 || y >= h)
		{
			for (int u = 0; u < c; u++)
				p_img_x[p + u * padded_tile_size] = 0;
		}
		else
		{
			for (int u = 0; u < c; u++)
				p_img_x[p + u * padded_tile_size] = itg[u * w * h + y * w + x];
		}
	}

	__syncthreads();

	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int x = blockIdx.x * blockDim.x + threadIdx.x;

	// compute convolution.
	for (int ki = 0; ki < k; ki++)
	{
		double res = 0;
		for (int ci = 0; ci < c; ci++)
		{
			for (int j = 0; j < fh; j++)
			{
				for (int i = 0; i < fw; i++)
				{
					double filtering = gpuf[ki * c * fh * fw + ci * fh * fw + (fh - 1 - j) * fw + (fw - 1 - i)];
					double inputs = p_img_x[ci * padded_tile_size + (threadIdx.y + j) * (blockDim.x + fw - 1) + threadIdx.x + i];
					res += inputs * filtering;
				}
			}
		}

		if (x < w && y < h)
			otg[ki * h * w + y * w + x] = res;
	}
}

#define check_cudnn(expression)                               \
	{                                                        \
		cudnnStatus_t curr = (expression);                   \
		if (curr != CUDNN_STATUS_SUCCESS)                    \
		{                                                    \
			printf("cuDNN error on line %d: %s\n", __LINE__, \
				   cudnnGetErrorString(curr));               \
			exit(EXIT_FAILURE);                              \
		}                                                    \
	}

int main(int argc, char *argv[])
{
	double *it, *ot;
	double *itg, *otg;
	double ConvolutionTime = 0;
	double CompTime;
	struct timespec start_time;
	struct timespec end_time;
	double *f;
	double *fT;
	double *fcuDNN;
	double *gpuf;
	double *gpufcuDNN;
	double filter[K][C][FH][FW];
	double filterCopy[K][C][FH][FW];

	/**
	 * c1 Convolution in CUDA
	 * Print the checksum as the total sum of the elements of O along all its dimensions
	 * Report the time to execute the CUDA kernel with the convolution,
	 */

	size_t OPTensor = K * H * W * sizeof(double);
	size_t IPTensor = C * H * W * sizeof(double);

	// Allocating space in CPU.
	it = (double *)malloc(IPTensor);
	ot = (double *)malloc(OPTensor);

	// Allocating space in GPU.
	cudaMalloc(&itg, IPTensor);
	cudaMalloc(&otg, OPTensor);

	// Steps for creating a filter of size K*C*FW*FH
	cudaMalloc(&gpuf, (K * C * FW * FH) * sizeof(double));
	cudaMalloc(&gpufcuDNN, (K * C * FW * FH) * sizeof(double));
	f = (double *)malloc((K * C * FW * FH) * sizeof(double));
	fT = (double *)malloc((K * C * FW * FH) * sizeof(double));
	fcuDNN = (double *)malloc((K * C * FW * FH) * sizeof(double));

	for (int k = 0; k < K; k++)
	{
		for (int c = 0; c < C; c++)
		{
			for (int j = 0; j < FH; j++)
			{
				for (int i = 0; i < FW; i++)
				{
					filter[k][c][j][i] = (c + k) * (i + j);
					fT[k * C * FH * FW + c * FH * FW + j * FW + i] = (c + k) * (i + j);
				}
			}
		}
	}
	for (int k = 0; k < K; k++)
	{
		for (int c = 0; c < C; c++)
		{
			for (int j = 0; j < FH; j++)
			{
				for (int i = 0; i < FW; i++)
					filterCopy[k][c][j][i] = filter[k][c][FH - 1 - j][FW - 1 - i];
			}
		}
	}
	for (int k = 0; k < K; k++)
	{
		for (int c = 0; c < C; c++)
		{
			for (int j = 0; j < FH; j++)
			{
				for (int i = 0; i < FW; i++)
					f[k * C * FH * FW + c * FH * FW + j * FW + i] = filterCopy[k][c][j][i];
			}
		}
	}
	for (int k = 0; k < K; k++)
	{
		for (int c = 0; c < C; c++)
		{
			for (int j = 0; j < FH; j++)
			{
				for (int i = 0; i < FW; i++)
					fcuDNN[k * C * FH * FW + c * FH * FW + j * FW + i] = filter[k][c][j][i];
			}
		}
	}
	// checksum
	double checksum = 0;
	for (int i = 0; i < (K * C * FH * FW); i++)
	{
		checksum += fcuDNN[i];
	}

	// Copying the filter cpu memory  to gpu.
	cudaMemcpy(gpuf, fT, (K * C * FH * FW) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(gpufcuDNN, fcuDNN, (K * C * FH * FW) * sizeof(double), cudaMemcpyHostToDevice);
	free(fT);
	free(fcuDNN);

	// Allocating space in CPU.
	it = (double *)malloc(IPTensor);
	ot = (double *)malloc(OPTensor);

	// Allocating space in GPU.
	cudaMalloc(&itg, IPTensor);
	cudaMalloc(&otg, OPTensor);

	// Row major ordering.
	load_image(H, W, C, it);

	// Time to kernel launch.
	clock_gettime(CLOCK_MONOTONIC, &start_time);
	convolution2DKernel<<<1, 256>>>(C, K, H, W, FH, FW, itg, gpuf, otg);
	cudaDeviceSynchronize();
	clock_gettime(CLOCK_MONOTONIC, &end_time);
	CompTime = convert_to_sec(&end_time) - convert_to_sec(&start_time);
	printf("C1_checksum: %.2f, C1_execution_time: %lfms\n", checksum, CompTime * 1000);

	/**
	 * c2 Tiled Convolution with CUDA
	 * Print the checksum as the sum of the elements of O along all its dimensions.
	 * Report the time to execute the CUDA kernel with the convolution
	 *
	 */

	// make space in CPU.
	it = (double *)malloc(IPTensor);
	ot = (double *)malloc(OPTensor);

	// make space in GPU.
	cudaMalloc(&itg, IPTensor);
	cudaMalloc(&otg, OPTensor);

	// Tile Width and Height.
	dim3 tileSize(TW, TH);
	dim3 numOfTiles((((W) + (TW)-1) / (TW)), (((H) + (TH)-1) / (TH)));

	// create filter
	cudaMalloc(&gpuf, (K * C * FW * FH) * sizeof(double));
	cudaMalloc(&gpufcuDNN, (K * C * FW * FH) * sizeof(double));
	f = (double *)malloc((K * C * FW * FH) * sizeof(double));
	fT = (double *)malloc((K * C * FW * FH) * sizeof(double));
	fcuDNN = (double *)malloc((K * C * FW * FH) * sizeof(double));

	// Filling the input filter. (2 ways of doing it)
	for (int k = 0; k < K; k++)
	{
		for (int c = 0; c < C; c++)
		{
			for (int j = 0; j < FH; j++)
			{
				for (int i = 0; i < FW; i++)
				{
					filter[k][c][j][i] = (c + k) * (i + j);
					fT[k * C * FH * FW + c * FH * FW + j * FW + i] = (c + k) * (i + j);
				}
			}
		}
	}
	//  Transposition.
	for (int k = 0; k < K; k++)
	{
		for (int c = 0; c < C; c++)
		{
			for (int j = 0; j < FH; j++)
			{
				for (int i = 0; i < FW; i++)
					filterCopy[k][c][j][i] = filter[k][c][FH - 1 - j][FW - 1 - i];
			}
		}
	}

	// Row order format.
	for (int k = 0; k < K; k++)
	{
		for (int c = 0; c < C; c++)
		{
			for (int j = 0; j < FH; j++)
			{
				for (int i = 0; i < FW; i++)
					f[k * C * FH * FW + c * FH * FW + j * FW + i] = filterCopy[k][c][j][i];
			}
		}
	}
	for (int k = 0; k < K; k++)
	{
		for (int c = 0; c < C; c++)
		{
			for (int j = 0; j < FH; j++)
			{
				for (int i = 0; i < FW; i++)
					fcuDNN[k * C * FH * FW + c * FH * FW + j * FW + i] = filter[k][c][j][i];
			}
		}
	}

	// compute checksum
	checksum = 0;
	for (int i = 0; i < (K * C * FH * FW); i++)
	{
		checksum += fcuDNN[i];
	}

	cudaMemcpy(gpuf, fT, (K * C * FH * FW) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(gpufcuDNN, fcuDNN, (K * C * FH * FW) * sizeof(double), cudaMemcpyHostToDevice);
	free(fT);
	free(fcuDNN);

	int shared_memory_size = C * (TW + FW - 1) * (TH + FH - 1) * sizeof(double);
	// Allocating space in CPU.
	it = (double *)malloc(IPTensor);
	ot = (double *)malloc(OPTensor);

	// Allocating space in GPU.
	cudaMalloc(&itg, IPTensor);
	cudaMalloc(&otg, OPTensor);

	// Row major ordering.
	load_image(H, W, C, it);

	// Time to kernel launch.
	clock_gettime(CLOCK_MONOTONIC, &start_time);
	// Function to call for kernel launch
	convolution2DKernel<<<numOfTiles, tileSize, shared_memory_size>>>(C, K, H, W, FH, FW, itg, gpuf, otg);
	cudaDeviceSynchronize();
	clock_gettime(CLOCK_MONOTONIC, &end_time);
	CompTime = convert_to_sec(&end_time) - convert_to_sec(&start_time);
	ConvolutionTime += CompTime;

	printf("C2_checksum: %.2f, C2_execution_time: %lfms\n", checksum, CompTime * 1000);

	return 0;
}
