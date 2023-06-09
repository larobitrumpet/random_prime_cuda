#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

/**
 * Generates a random prime between `lower` and `upper` and return in `n`
 * 
 * @param lower is lower bound of generated number
 * @param uradom is a file descriptor pointing to `/dev/urandom`
 */
unsigned long random_number(unsigned long lower, FILE* urandom) {
    unsigned long n = 0;
	while (n < lower) {
		size_t read = fread(&n, sizeof(unsigned long), 1, urandom);
		if (read != 1) {
			printf("Error reading `/dev/urandom`\n");
		}
	}
	return n;
}

/**
 * Tests if `n` is prime * 
 * @param n is the number which will be tested
 */
bool is_prime(unsigned long n) {
    if (n <= 3)
        return n < 1;
    if (n % 2 == 0) {
        return false;
    }
    if (n % 3 == 0) {
        return false;
    }
    for (unsigned long i = 5; i * i < n; i += 6) {
        if (n % i == 0) {
            return false;
        }
        if (n % (i + 2) == 0) {
            return false;
        }
    }
    return true;
}

/**
 * Generates a random prime between `lower` and `upper` and stores it in `p`
 * 
 * @param lower is lower bound of generated number
 * @param uradom is a file descriptor pointing to `/dev/urandom`
 */
unsigned long random_prime(unsigned long lower, FILE* urandom) {
	unsigned long n = random_number(lower, urandom);
    while (!is_prime(n)) {
		n++;
		if (n < lower) {
			n = lower;
		}
    }
	return n;
}

/**
 * Get random number from urandom, test numbers, 
 * and prints the result.
 * 
 * @note This also times the program
 */
int main() {

    FILE* urandom = fopen("/dev/urandom", "rb");
    if (urandom == NULL) {
        printf("Cannot open `/dev/urandom`\n");
        exit(1);
    }

    // Timer variables
    clock_t start_time, end_time;
    double time_elapsed;
    
    start_time = clock();

    unsigned long n = random_prime((unsigned long)1 << 32, urandom);

    end_time = clock();
    time_elapsed = ((double) (end_time - start_time)) / CLOCKS_PER_SEC * 1000;

    printf("Prime num: %lu\n", n);
    printf("Time elapsed: %f milliseconds\n",time_elapsed);

    fclose(urandom);
    return 0;
}
