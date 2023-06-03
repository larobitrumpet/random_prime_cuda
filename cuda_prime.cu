
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <cuda.h>
/*
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
*/
// Tests if `n` is prime
__global__ void is_prime(const unsigned long n[], bool p[]) {

    int  my_idx = threadIdx.x;
    unsigned long my_n = n[my_idx];

    if (my_n <= 3)
        p[my_idx] = my_n<1;
        return;
         //return n < 1;
    if (my_n % 2 == 0) {
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
/*
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
}*/

// From the command line, pass in the number of bits
// for the generated prime, defaults to 64 bits.
int main() {
    FILE* urandom = fopen("/dev/urandom", "rb");
    if (urandom == NULL) {
        printf("Cannot open `/dev/urandom`\n");
        exit(1);
    }
    int blk_ct = 1;
    int th_per_blk = 32;
    unsigned long rand;
    unsigned long* n;
    bool* is_p;

    size_t read = fread(&rand, sizeof(unsigned long), 1, urandom);
    if (read != 1) {
       printf("Error reading `/dev/urandom`\n");
    }

    cudaMallocManaged(&n, th_per_blk*sizeof(unsigned long));
    cudaMallocManaged(&is_p, th_per_blk*sizeof(bool));
    rand = 1148768376218426693;
    for(int i=0; i<th_per_blk; i+=1){
            n[i] = i+rand;
    }

    is_prime<<<blk_ct, th_per_blk>>>(n, is_p);
    cudaDeviceSynchronize();

   for(int i=0; i<th_per_blk; i++)
           if (is_p[i])
                   printf("%lu\n", n[i]);

    fclose(urandom);
    cudaFree(n);
    return 0;
}
