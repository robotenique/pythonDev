import numpy as np
from functools import lru_cache
from dp_timer import timer_wrap

def main():
    N_ITER = 30
    FIB_ARG = 60
    timer_wrap(memo_fib,  (FIB_ARG), N_ITER)
    timer_wrap(lru_fib,   (FIB_ARG), N_ITER)
    timer_wrap(iter_fib,  (FIB_ARG), N_ITER)
    timer_wrap(gen_fib,   (FIB_ARG), N_ITER)
    timer_wrap(naive_fib, (FIB_ARG), N_ITER)


# Naive Fibonacci impl
def naive_fib(n):
    if n <= 1:
        return n
    return naive_fib(n - 1) + naive_fib(n - 2)

# Memoized Fibonacci impl
def memo_fib(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    fib_table = np.zeros(n + 1, dtype=int) - 1
    fib_table[0] = 0
    fib_table[1] = 1

    def memo_fib_inner(n):
        if fib_table[n] >= 0:
            return fib_table[n]
        fib_table[n] = memo_fib_inner(n - 1) + memo_fib_inner(n - 2)
        return fib_table[n]

    return memo_fib_inner(n)

# Memoization using python functools - causes max recursion depth
def lru_fib(n):

    @lru_cache(maxsize=n+1)
    def lru_fib_inner(n):
        if n <= 1:
            return n
        return lru_fib_inner(n - 1) + lru_fib_inner(n - 2)

    return lru_fib_inner(n)

# Iterative fibonacci
def iter_fib(n):
    fib_table = np.zeros(n + 1, dtype=int) - 1
    fib_table[0] = 0
    fib_table[1] = 1
    for i in range(2, n + 1):
        fib_table[i] = fib_table[i - 1] + fib_table[i - 2]
    return fib_table[n]


# Generator fib approach
def gen_fib(n):
    if n <= 1:
        return n

    def gen_fib_inner():
        a = 0
        b = 1
        while True:
            yield a
            a, b = b, a + b

    fibber = gen_fib_inner()

    [next(fibber) for _ in range(n)]

    return next(fibber)




if __name__ == '__main__':
    main()
