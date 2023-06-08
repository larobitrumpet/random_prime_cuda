#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <cuda.h>

// Generates a random unsigned long greater than `lower`
// `uradom` is a file descriptor pointing to `/dev/urandom`
unsigned long random_number(unsigned long lower, FILE* urandom) {
	unsigned long n = 0;
	while (n < lower) {
		size_t read = fread(&n, sizeof(unsigned long), 1, urandom);
		if (read != 1) {
			printf("Error reading `/dev/urandom`\n");
		}
	}
	if (n % 2 == 0) {
		n++;
	}
	return n;
}

__global__ void is_prime_part(const unsigned long n, bool p[], const unsigned long block_size) {
	int my_idx = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned long lower = block_size * my_idx + 5;
	unsigned long upper = block_size * (my_idx + 1) + 5;
	p[my_idx] = true;
	for (unsigned long i = lower; i < upper; i += 6) {
		if (n % i == 0) {
			p[my_idx] = false;
			return;
		}
		if (n % (i + 2) == 0) {
			p[my_idx] = false;
			return;
		}
	}
}

// Tests if `n` is prime
bool is_prime(const unsigned long n, int blk_ct, int th_per_blk) {
	if (n <= 3) {
		return n < 1;
	}
	if (n % 2 == 0) {
		return false;
	}
	if (n % 3 == 0) {
		return false;
	}

	unsigned long block_size = sqrt(n) / (blk_ct * th_per_blk);
	block_size += 6 - (block_size % 6);
	bool* p;
	cudaMallocManaged(&p, blk_ct * th_per_blk * sizeof(bool));

	is_prime_part<<<blk_ct, th_per_blk>>>(n, p, block_size);

	cudaDeviceSynchronize();
	for (int i = 0; i < blk_ct * th_per_blk; i++) {
		if (p[i] == false) {
			return false;
		}
	}

	cudaFree(p);
	return true;
}

int main() {
	FILE* urandom = fopen("/dev/urandom", "rb");
	if (urandom == NULL) {
		printf("Cannot open `/dev/urandom`\n");
		exit(1);
	}
	
	int blk_ct = 128;
	int th_per_blk = 512;
	unsigned long lower = (unsigned long)1 << 32 + 1;
	unsigned long rand = random_number(lower, urandom);

	while (true) {
		if (is_prime(rand, blk_ct, th_per_blk)) {
			printf("%lu\n", rand);
			break;
		}
		rand += 2;
		if (rand < lower) {
			rand = lower;
		}
	}

	fclose(urandom);
	return 0;
}
