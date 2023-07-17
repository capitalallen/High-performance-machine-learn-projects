#include <stdio.h>
#include <time.h>
#include <assert.h>
#include <cudnn.h>
// Declaring constants.

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

	double cuDNNtime = 0;
	double CompTime;
	struct timespec start_time;
	struct timespec end_time;

	double *gpuf;
	double *gpufcuDNN;

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
	// C3 - Convolution with cuDNN.
	// Reference: http://www.goldsborough.me/cuda/ml/cudnn/c++/2017/10/01/14-37-23-convolutions_with_cudnn/

	cudnnHandle_t cudnn;
	cudnnCreate(&cudnn);

	cudnnTensorDescriptor_t input_descriptor;
	check_cudnn(cudnnCreateTensorDescriptor(&input_descriptor));
	check_cudnn(cudnnSetTensor4dDescriptor(input_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, C, H, W));

	cudnnTensorDescriptor_t output_descriptor;
	check_cudnn(cudnnCreateTensorDescriptor(&output_descriptor));
	check_cudnn(cudnnSetTensor4dDescriptor(output_descriptor, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, K, H, W));

	cudnnFilterDescriptor_t kernel_descriptor;
	check_cudnn(cudnnCreateFilterDescriptor(&kernel_descriptor));
	check_cudnn(cudnnSetFilter4dDescriptor(kernel_descriptor, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, K, C, FH, FW));

	cudnnConvolutionDescriptor_t convolution_descriptor;
	check_cudnn(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
	check_cudnn(cudnnSetConvolution2dDescriptor(convolution_descriptor,
											   /*pad_height=*/1,
											   /*pad_width=*/1,
											   /*vertical_stride=*/1,
											   /*horizontal_stride=*/1,
											   /*dilation_height=*/1,
											   /*dilation_width=*/1,
											   /*mode=*/CUDNN_CONVOLUTION,
											   /*computeType=*/CUDNN_DATA_DOUBLE));

	cudnnConvolutionFwdAlgo_t convolution_algorithm;
    cudnnGetConvolutionForwardAlgorithm(cudnn,
                                        input_descriptor,
                                        kernel_descriptor,
                                        convolution_descriptor,
                                        output_descriptor,
                                        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                        /*memoryLimitInBytes=*/0,
                                        &convolution_algorithm);

	size_t workspace_bytes = 0;
	check_cudnn(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
													   input_descriptor,
													   kernel_descriptor,
													   convolution_descriptor,
													   output_descriptor,
													   convolution_algorithm,
													   &workspace_bytes));

	void *d_workspace;
	cudaMalloc(&d_workspace, workspace_bytes);

	double alpha = 1, beta = 0.0;
	it = (double *)malloc(IPTensor);
	ot = (double *)malloc(OPTensor);
	cudaMalloc(&itg, IPTensor);
	cudaMalloc(&otg, OPTensor);
	load_image(H, W, C, it);
	clock_gettime(CLOCK_MONOTONIC, &start_time);
	check_cudnn(cudnnConvolutionForward(cudnn,
									   &alpha,
									   input_descriptor,
									   itg,
									   kernel_descriptor,
									   gpufcuDNN,
									   convolution_descriptor,
									   convolution_algorithm,
									   d_workspace,
									   workspace_bytes,
									   &beta,
									   output_descriptor,
									   otg));
	cudaDeviceSynchronize();
	clock_gettime(CLOCK_MONOTONIC, &end_time);
	CompTime = convert_to_sec(&end_time) - convert_to_sec(&start_time);
	cuDNNtime += CompTime;


	// Output tensor checksum.
	double checksum = 0;
	for (int i = 0; i < (H * K * W); i++){
		checksum += ot[i];
	}
	printf("C3_checksum: %.2f, C3_execution_time: %lfms\n", checksum, CompTime*1000);


	return 0;
}
