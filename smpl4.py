'''
2.1 - Numpy Array library
'''
#!/usr/bin/env
import numpy as np

def main():
    ''''
    # 1) Create two vectors 'a' and 'b', and sum them.
    # Standart
    n = 3
    a = [x**2 for x in range(1, n + 1)]
    b = [x**3 for x in range(1, n + 1)]
    v = plainVectorAddition(a, b)
    # Numpy
    n = 3
    a = np.arange(1, n + 1) ** 2
    b = np.arange(1, n + 1) ** 3
    v = npSum(a, b)
    '''
    print(v)


def plainSum(a, b):
    '''
    Sum two vectors using standart python functions
    '''
    return [i + j for i,j in zip(a,b)]

def npSum(a, b):
    return a + b



if __name__ == '__main__':
    main()
