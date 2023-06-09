#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>


/*
THIS TESTS RUNTIME WITH NUMBER 11058056269920516451
COUNT THE TIME FOR USING ABOVE SEED TO GENERATES A PRIME NUMBER
*/


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
unsigned long random_prime(unsigned long lower) {
	unsigned long n = 11058056269920516451U;
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

    // Timer variables
    clock_t start_time, end_time;
    double time_elapsed;
    
    start_time = clock();

    unsigned long n = random_prime((unsigned long)1 << 32);

    end_time = clock();

    time_elapsed = ((double) (end_time - start_time)) / CLOCKS_PER_SEC * 1000;

    printf("Prime num: %lu\n", n);
    printf("Time elapsed: %f milliseconds\n",time_elapsed);

    return 0;
}
