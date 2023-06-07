#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <cuda.h>

// Generates a random number between `lower` and `upper` and stores it in `n`
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

// Tests if `n` is prime
__global__ void is_prime(const unsigned long n, bool p[]) {
	int  my_idx = threadIdx.x;
	unsigned long my_n = n + my_idx * 2;

	if (my_n <= 3) {
		p[my_idx] = my_n < 1;
		return;
	}
	if (my_n % 2 == 0) {
	printf("%lu is even\n", my_n);
		p[my_idx] = false;
		return;
	}
	if (my_n % 3 == 0) {
		p[my_idx] = false;
		return;
	}
	for (unsigned long i = 5; i * i < my_n; i += 6) {
		if (my_n % i == 0) {
			p[my_idx] = false;
			return;
		}
		if (my_n % (i + 2) == 0) {
			p[my_idx] = false;
			return;
		}
	}
	p[my_idx] = true;
	return;
}

int main() {
	FILE* urandom = fopen("/dev/urandom", "rb");
	if (urandom == NULL) {
		printf("Cannot open `/dev/urandom`\n");
		exit(1);
	}
	
	int blk_ct = 1;
	int th_per_blk = 32;
	unsigned long rand = random_number((unsigned long)1<<32, urandom);
	bool* is_p;
	bool noprime = true;

	cudaMallocManaged(&is_p, th_per_blk*sizeof(bool));

	while (noprime) {
		is_prime<<<blk_ct, th_per_blk>>>(rand, is_p);
		cudaDeviceSynchronize();

		for(int i=0; i<th_per_blk; i++) {
			if (is_p[i]) {
				printf("%d: %lu\n", is_p[i], rand+i*2);
				noprime = false;
				break;
			}
		}
	}

	fclose(urandom);
	cudaFree(is_p);
	return 0;
}
