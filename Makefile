CC := gcc
CFLAGS := -Wall -Wextra -Werror

ifneq ($(D),1)
CFLAGS += -O2
else
CFLAGS += -g
endif

all: random_prime cuda_prime cuda_prime_2

random_prime: random_prime.c
        $(CC) $(CFLAGS) random_prime.c -o random_prime
cuda_prime: cuda_prime.cu
        nvcc cuda_prime.cu -o cuda_prime
cuda_prime_2: cuda_prime_2.cu
        nvcc cuda_prime_2.cu -o cuda_prime_2
clean:
        rm random_prime cuda_prime cuda_prime_2

