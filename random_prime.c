#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

// The base to print the prime number in
#define BASE 10

// Generates a random number between `lower` and `upper` and stores it in `n`
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

// Tests if `n` is prime
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

// Generates a random prime between `lower` and `upper` and sores it in `p`
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

// From the command line, pass in the number of bits
// for the generated prime, defaults to 64 bits.
int main() {
    FILE* urandom = fopen("/dev/urandom", "rb");
    if (urandom == NULL) {
        printf("Cannot open `/dev/urandom`\n");
        exit(1);
    }

    unsigned long n = random_prime((unsigned long)1 << 32, urandom);
    printf("%lu\n", n);

    fclose(urandom);
    return 0;
}
