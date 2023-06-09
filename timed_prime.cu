#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
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
 * @note This is 32 threads cuda data processing
 */
__global__ void is_prime(const unsigned long n, bool p[])
{
	int my_idx = threadIdx.x;
	unsigned long my_n = n + my_idx * 2;

	if (my_n <= 3)
	{
		p[my_idx] = my_n < 1;
		return;
	}
	if (my_n % 2 == 0)
	{
		printf("%lu is even\n", my_n);
		p[my_idx] = false;
		return;
	}
	if (my_n % 3 == 0)
	{
		p[my_idx] = false;
		return;
	}
	for (unsigned long i = 5; i * i < my_n; i += 6)
	{
		if (my_n % i == 0)
		{
			p[my_idx] = false;
			return;
		}
		if (my_n % (i + 2) == 0)
		{
			p[my_idx] = false;
			return;
		}
	}
	p[my_idx] = true;
	return;
}

/**
 * Get random number from urandom, initial cuda kernel,
 * and prints the result.
 *
 * @note This also times the threads
 */
int main()
{

	// Initially, 128 block * 512 thread per block = 65536 threads
	int blk_ct = 1;
	int th_per_blk = 32;
	unsigned long rand = 11058056269920516451;
	bool *is_p;
	bool noprime = true;

	cudaMallocManaged(&is_p, th_per_blk * sizeof(bool));

	// Time varaibles
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Start timer here
	cudaEventRecord(start);

	while (noprime)
	{
		is_prime<<<blk_ct, th_per_blk>>>(rand, is_p);
		cudaDeviceSynchronize();

		for (int i = 0; i < th_per_blk; i++)
		{
			if (is_p[i])
			{
				printf("\nPrime num(%d): %lu\n", is_p[i], rand + i * 2);
				noprime = false;
				break;
			}
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

	cudaFree(is_p);
	return 0;
}
