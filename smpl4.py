'''
2.1 - Numpy Array library
 - Sum of two vectors - 
 
 Usage: python smpl4.py n

 Where 'n' specifies the size of the vector.
 The program make performance comparisons and print the results.

'''
#!/usr/bin/env
import numpy as np
import sys
from datetime import datetime

def main():
    size = int(sys.argv[1])
    start = datetime.now()
    v = plainSum(size)
    delta = datetime.now() - start
    print("plainSum() - elapsed time: ",delta)
    start = datetime.now()
    v = npSum(size)
    delta = datetime.now() - start
    print("npSum() - elapsed time: ",delta)
    ''''
    # 1) Create two vectors 'a' and 'b', and sum them.
    # Standart
    size = int(sys.argv[1])   
    v = plainVectorAddition(a, b)    
    # Numpy
    size = int(sys.argv[1])
    v = npSum(a, b)
    '''
    #print(v)



def plainSum(n):
    '''
    Sum two vectors using standart python functions
    '''
    a = [x**2 for x in range(1, n + 1)]
    b = [x**3 for x in range(1, n + 1)]
    return [i + j for i,j in zip(a,b)]

def npSum(n):
    '''
    Sum two vectors using numpy functions
    '''
    a = np.arange(1, n + 1) ** 2
    b = np.arange(1, n + 1) ** 3
    return a + b


if __name__ == '__main__':
    main()
