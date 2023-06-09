CC := gcc
CFLAGS := -Wall -Wextra -Werror

ifneq ($(D),1)
CFLAGS += -O2
else
CFLAGS += -g
endif

all: random_prime cuda_prime cuda_prime_2 timed_random timed_prime timed_prime_2

random_prime: random_prime.c
        $(CC) $(CFLAGS) random_prime.c -o random_prime
cuda_prime: cuda_prime.cu
        nvcc cuda_prime.cu -o cuda_prime
cuda_prime_2: cuda_prime_2.cu
        nvcc cuda_prime_2.cu -o cuda_prime_2
timed_random: timed_random.c
        $(CC) $(CFLAGS) timed_random.c -o timed_random
timed_prime: timed_prime.cu
        nvcc timed_prime.cu -o timed_prime
timed_prime_2: timed_prime_2.cu
        nvcc timed_prime_2.cu -o timed_prime_2
clean:
        rm random_prime cuda_prime cuda_prime_2 timed_random timed_prime timed_prime_2

