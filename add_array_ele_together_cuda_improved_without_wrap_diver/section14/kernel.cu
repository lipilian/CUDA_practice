#include <stdio.h>
#include <stdlib.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "common.h"
#include "cuda_common.cuh"

__global__ void reduction_neighbored_paris_improved(int* input, int* temp, int size) {
	int tid = threadIdx.x;
	int gid = blockDim.x * blockIdx.x + tid;

	//local data block pointer
	//int* i_data = input + blockDim.x * blockIdx.x;

	if (gid > size)
		return;

	for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
		//int index = 2 * offset * tid;

		//if (index < blockDim.x) {
		//	i_data[index] += i_data[index + offset];
		//}
		if (tid < offset) {
			input[gid] += input[gid + offset];
		}
		__syncthreads();
	}
	if (tid == 0) {
		temp[blockIdx.x] = input[gid];
	}
}

int main(int argc, char** argv)
{	
	printf("Running neighbored paris reduction kernel \n");

	int size = 1 << 27; // 128Mb of data
	int byte_size = size * sizeof(int);
	int block_size = 128;

	int* h_input, * h_ref;
	h_input = (int*)malloc(byte_size);
	initialize(h_input, size, INIT_RANDOM);

	//GET CPU RESULT
	int cpu_result = reduction_cpu(h_input, size);

	dim3 block(block_size);
	dim3 grid(size / block.x);

	printf("Kernel lauch parameters | grid :%d, block: %d", grid.x, block.x);

	int temp_array_byte_size = sizeof(int) * grid.x;

	h_ref = (int*)malloc(temp_array_byte_size);

	int* d_input, * d_temp;

	gpuErrchk(cudaMalloc((void**)&d_input, byte_size));
	gpuErrchk(cudaMalloc((void**)&d_temp, temp_array_byte_size));

	cudaMemset(d_temp, 0, temp_array_byte_size);
	cudaMemcpy(d_input, h_input, byte_size, cudaMemcpyHostToDevice);

	reduction_neighbored_paris_improved << <grid, block >> > (d_input, d_temp, size);

	gpuErrchk(cudaDeviceSynchronize());

	cudaMemcpy(h_ref, d_temp, temp_array_byte_size, cudaMemcpyDeviceToHost);

	int gpu_result = 0;

	for (int i = 0; i < grid.x; i++) {
		gpu_result += h_ref[i];
	}

	compare_results(gpu_result, cpu_result);

	cudaFree(d_temp);
	cudaFree(d_input);

	free(h_ref);
	free(h_input);
	
	gpuErrchk(cudaDeviceReset());
	return 0;
}