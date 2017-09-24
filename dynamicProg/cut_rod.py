import math
import numpy as np
from functools import lru_cache
from dp_timer import timer_wrap

def main():
    global P_LIST
    N_ITER = 300
    NUM_RODS = 100
    MAX_PRICE = 20
    MIN_PRICE = 1
    SIZE = 50

    # The price array
    P_LIST = (np.random.rand(NUM_RODS + 1) * MAX_PRICE + MIN_PRICE).astype(int)

    timer_wrap(memo_cut_rod, (SIZE, P_LIST), niter = N_ITER)
    timer_wrap(cached_cut_rod, (SIZE, P_LIST), niter = N_ITER)
    timer_wrap(iter_cut_rod, (SIZE, P_LIST), niter = N_ITER)
    timer_wrap(naive_cut_rod,  (SIZE, P_LIST), niter = N_ITER)


# Naive implementation of cutting rod - Exponential time...
def naive_cut_rod(n, p):
    if n == 0:
        return 0
    q = -math.inf
    for i in range(1, n + 1):
        q = max(q, p[i] + naive_cut_rod(n - i, p))
    return q

# memoized cutting rod
def memo_cut_rod(n, p):
    max_costT = np.zeros(n + 1) - math.inf
    cut_sizes = np.zeros(n + 1, dtype=int) - math.inf
    max_costT[0] = 0

    def memo_inner(n):
        if max_costT[n] >= 0:
            return max_costT[n]
        q = -math.inf
        for i in range(1, n + 1):
            calculated = p[i] + memo_inner(n - i)
            if calculated > q:
                q = calculated
                cut_sizes[n] = i
        max_costT[n] = q
        return int(q)

    return memo_inner(n), cut_sizes

# Iterative cutting rod
def iter_cut_rod(n, p):
    max_costT = np.zeros(n + 1) - math.inf
    cut_sizes = np.zeros(n + 1, dtype=int) - math.inf
    max_costT[0] = 0

    for j in range(1, n + 1):
        q = -math.inf

        for i in range(1, j + 1):
            calculated = p[i] + max_costT[j - i]
            if calculated > q:
                q = calculated
                cut_sizes[j] = i

        max_costT[j] = q

    return int(max_costT[j]), cut_sizes


# Using python lru_cache: memoized function
def cached_cut_rod(n, p):
    cut_sizes = np.zeros(n + 1, dtype=int) - math.inf
    @lru_cache(maxsize=n+1)
    def cached_inner(n):
        if n == 0:
            return 0
        q = -math.inf
        for i in range(1, n + 1):
            calculated = p[i] + cached_inner(n - i)
            if calculated > q:
                q = calculated
                cut_sizes[n] = i
        return int(q)

    return cached_inner(n), cut_sizes

# Recover the cut sizes
def recover_cuts(cut_sizes, n):
    n = int(n)
    if n == 0 or cut_sizes[n] == n or n == -math.inf:
        return []
    return [int(cut_sizes[n])] + recover_cuts(cut_sizes, n - cut_sizes[n])



if __name__ == '__main__':
    main()
