#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

//for random intialize
#include <stdlib.h>
#include <time.h>

//for memset
#include <cstring>

//Compare array
#include "common.h"

//Error Check for GPU
#include "cuda_common.cuh"

__global__ void sum_array_gpu(int* a, int* b, int* c, int size) {
	int gid = blockIdx.x * blockDim.x + threadIdx.x; // calculate the gid for each element in gpu

	if (gid < size)
	{
		c[gid] = a[gid] + b[gid];
	}
}

void sum_array_cpu(int* a, int* b, int* c, int size) {
	for(int i = 0; i < size; i++)
	{
		c[i] = a[i] + b[i];
	}
}

int main() {
	int size = 1 << 25;
	int block_size = 128;
	cudaError error;

	int NO_BYTES = size * sizeof(int);
	int* h_a, * h_b, * gpu_results, *h_c;


	//allocate memory
	h_a = (int*)malloc(NO_BYTES);
	h_b = (int*)malloc(NO_BYTES);
	gpu_results = (int*)malloc(NO_BYTES);
	h_c = (int*)malloc(NO_BYTES);

	//intialze host value
	time_t t;
	srand((unsigned)time(&t));
	for (int i = 0; i < size; i++) {
		h_a[i] = (int)(rand() & 0xFF);
	}
	for (int i = 0; i < size; i++) {
		h_b[i] = (int)(rand() & 0xFF);
	}
	
	//summationo in CPU
	clock_t cpu_start, cpu_end;
	cpu_start = clock();
	sum_array_cpu(h_a, h_b, h_c, size);
	cpu_end = clock();




	memset(gpu_results, 0, NO_BYTES);

	//device pointer

	int* d_a, * d_b, * d_c;
	gpuErrchk(cudaMalloc((int**)& d_a, NO_BYTES));
	
	gpuErrchk(cudaMalloc((int**)& d_b, NO_BYTES));
	
	gpuErrchk(cudaMalloc((int**)& d_c, NO_BYTES));

	clock_t htod_start, htod_end;
	htod_start = clock();
	cudaMemcpy(d_a, h_a, NO_BYTES, cudaMemcpyHostToDevice); // transfer memory from host to gpu
	cudaMemcpy(d_b, h_b, NO_BYTES, cudaMemcpyHostToDevice);
	htod_end = clock();
	//lauching the grid
	dim3 block(block_size); // define blcck
	dim3 grid((size / block.x) + 1); // define grid

	clock_t gpu_start, gpu_end;
	gpu_start = clock();
	sum_array_gpu << <grid, block >> > (d_a, d_b, d_c, size); // call function in gpu
	cudaDeviceSynchronize(); // hold the code until cuda give feed back
	gpu_end = clock();

	clock_t dtoh_start, dtoh_end;
	dtoh_start = clock();
	cudaMemcpy(gpu_results, d_c, NO_BYTES, cudaMemcpyDeviceToHost);
	dtoh_end = clock();

	//array comparison
	compare_arrays(gpu_results,h_c,size);


	printf("Sum array cpu execution time : %4.6f \n", (double)((double)(cpu_end - cpu_start) / CLOCKS_PER_SEC));
	printf("Sum array gpu execution time : %4.6f \n", (double)((double)(gpu_end - gpu_start) / CLOCKS_PER_SEC));
	printf("Device to Host memory transfer time : %4.6f \n", (double)((double)(dtoh_end - dtoh_start) / CLOCKS_PER_SEC));
	printf("Host to Device memory transfer time : %4.6f \n", (double)((double)(htod_end - htod_start) / CLOCKS_PER_SEC));
	printf("Total GPU time : %4.6f \n", (double)((double)(dtoh_end - htod_start) / CLOCKS_PER_SEC));

	cudaFree(d_c);
	cudaFree(d_a);
	cudaFree(d_b);

	free(h_b);
	free(h_a);
	free(gpu_results);

	cudaDeviceReset();
	return 0;
}