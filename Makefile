CC := gcc
CFLAGS := -Wall -Wextra -Werror

ifneq ($(D),1)
CFLAGS += -O2
else
CFLAGS += -g
endif

random_prime: random_prime.c
	$(CC) $(CFLAGS) random_prime.c -o random_prime

clean:
	rm random_prime
