#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <cuda.h>

/*
THIS TESTS RUNTIME WITH NUMBER 11058056269920516451
COUNT THE TIME FOR USING ABOVE SEED TO GENERATES A PRIME NUMBER
*/


/**
 * Tests if `n` is prime and stores the result in `p[threadIdx.x]`
 *
 * @param n is unsigned long in that will be tested
 * @param p is a boolean array that stores t/f info
 * @param block_size is clearly the block size
 * @note This is cuda function
 */
__global__ void is_prime_part(const unsigned long n, bool p[], const unsigned long block_size)
{
	int my_idx = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned long lower = block_size * my_idx + 5;
	unsigned long upper = block_size * (my_idx + 1) + 5;
	p[my_idx] = true;
	for (unsigned long i = lower; i < upper; i += 6)
	{
		if (n % i == 0)
		{
			p[my_idx] = false;
			return;
		}
		if (n % (i + 2) == 0)
		{
			p[my_idx] = false;
			return;
		}
	}
}

/**
 * Tests if `n` is prime
 *
 * @param n is unsigned long in that will be tested
 * @param blk_ct is the block number of cuda kernel
 * @param th_per_blk is the thread number of each block
 * @note This is also the function which calls __global__
 */
bool is_prime(const unsigned long n, int blk_ct, int th_per_blk)
{

	if (n <= 3)
	{
		return n < 1;
	}
	if (n % 2 == 0)
	{
		return false;
	}
	if (n % 3 == 0)
	{
		return false;
	}

	unsigned long block_size = sqrt(n) / (blk_ct * th_per_blk);
	block_size += 6 - (block_size % 6);
	bool *p;
	cudaMallocManaged(&p, blk_ct * th_per_blk * sizeof(bool));

	is_prime_part<<<blk_ct, th_per_blk>>>(n, p, block_size);

	cudaDeviceSynchronize();
	for (int i = 0; i < blk_ct * th_per_blk; i++)
	{
		if (p[i] == false)
		{

			cudaFree(p);
			return false;
		}
	}

	cudaFree(p);
	return true;
}

/**
 * Get random number from urandom, initial cuda kernel,
 * and prints the result.
 *
 * @note This also times the threads
 */
int main(int argc, char *argv[])
{

	// Initially, 128 block * 512 thread per block = 65536 threads
	int blk_ct = 128;
	int th_per_blk = 512;
	unsigned long lower = (unsigned long)1 << 32 + 1;
	unsigned long rand = 11058056269920516451;

	// Check if the user has provided values
	if (argc >= 2)
	{
		blk_ct = strtol(argv[1], NULL, 10);
	}
	if (argc >= 3)
	{
		th_per_blk = strtol(argv[2], NULL, 10);
	}

	// Time varaibles
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Start timer here
	cudaEventRecord(start);

	while (true)
	{
		if (is_prime(rand, blk_ct, th_per_blk))
		{
			printf("\nBlk: %d, threads_per_block: %d", blk_ct, th_per_blk);
			printf("\nPrime num: %lu\n", rand);
			break;
		}
		rand += 2;
		if (rand < lower)
		{
			rand = lower;
		}
	}

	// Stop timer
	cudaEventRecord(stop);

	// Wait for the stop event to complete
	cudaEventSynchronize(stop);

	// Show time
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Time elapsed: %f milliseconds\n", milliseconds);


	//fclose(urandom); not used in this case
	return 0;
}
