'''
2.1 - Numpy Array library
'''
import numpy as np

def main():
    n = 10
    a = [x**2 for x in range(n)]
    b = [x**3 for x in range(n)]
    v = plainVectorAddition(a, b)
    print(v)


def plainVectorAddition(a, b):
    v = list()
    for i,j,k in zip(a, b, len(a)):
        v[k] = i + j
    return v

if __name__ == '__main__':
    main()
