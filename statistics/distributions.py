'''
- Probability distributions -
Implementation of the calculation of the most used discrete
probability distribution in statistics!
'''
import numpy as np
import matplotlib as plt
import math as m
from fractions import Fraction
from functools import reduce

def main():
    lenProb = 6
    menu = [[] for x in range(lenProb + 1)]
    menu[0] = [
        "--- Welcome to the ProbDistributions =) ---\n",
        "You may choose one of the options below: \n",
        "[1] - Binomial Distribution\n",
        "[2] - Poisson Distribution\n"
        "[3] - Geometric Distribution\n"
        "[4] - Negative Binomial Distribution\n"
        "[5] - Hipergeometric Distribution\n"
    ]
    menu[1] = [
        "\n---BINOMIAL DISTRIBUTION: k sucesses in n tries, with probability p (each try) ---",
        "n",
        "p",
        "k"
    ]
    menu[2] = [
        "\n---POISSON DISTRIBUTION: k sucesses in a given time with rate λ ---",
        "k",
        "λ",
    ]
    menu[3] = [
        "\n---GEOMETRIC DISTRIBUTION: success in the k-th try, with probability p (each try)  ---",
        "k",
        "p",
    ]
    menu[4] = [
        "\n---NEGATIVE BINOMIAL DISTRIBUTION: number of tries until the k-th sucess, each with probability p (each try) ---",
        "k",
        "p",
        "x"
    ]
    menu[5] = [
        "\n---HIPERGEOMETRIC DISTRIBUTION: The probability of k successes in n draws, from a population with \n \
        size N, with total of K successes in the population ---",
        "k",
        "n",
        "N",
        "K"
    ]
    readIn = lambda x: float(input(x + " = "))
    probDistr = [0, binomial, poisson, geometric, negativeBinomial, hipergeometric]
    try:
        print("".join(menu[0]))
        io = int(input("==> "))
        if io < 1 or io >= lenProb: raise ValueError()
        print(menu[io][0])
        args = list(map(readIn, (menu[io][i] for i in range(1, len(menu[io])))))
        prob = probDistr[io](args)
        print("The " + probDistr[io].__name__ + " probability is " + str(round(prob, 3)) + "!")
    except ValueError:
        print("You didn't chose a correct input, now i'll kill myself! ;(")
        exit()

def binCoeff(n, x):
    '''
    Calculates the Binomial coefficient in the form nCx
    '''
    # Calculate the binomial coefficient
    binC = reduce(lambda x,y: x*y, (Fraction(n - i, i + 1) for i in range(x)), 1)
    return(int(binC))

def binomial(*args):
    try:
        n, p, k = int(args[0][0]), args[0][1], int(args[0][2])
        if p < 0 or p > 1:
            raise ValueError("Wrong probability value ("+str(p)+") !")
        return(binCoeff(n, k)*(p**k)*((1- p)**(n - k)))
    except ValueError as err:
        print(";( Error: ", err)
        exit()

def poisson(*args):
    try:
        k, rate = int(args[0][0]), args[0][1]
        if k > 20000:
            raise ValueError("Whoa, go do this somewhere else! >:(  !")
        return((m.exp(-rate)*(rate**k))/m.factorial(k))
    except ValueError as err:
        print(err)
        exit()

def geometric(*args):
    try:
        k, p = int(args[0][0]), args[0][1]
        if p < 0 or p > 1 or k < 0:
            raise ValueError("Wrong probability value ("+str(p)+") !")
        return(p*((1-p)**(k - 1)))
    except ValueError as err:
        print("ERROR:", err)
        exit()

def negativeBinomial(*args):
    try:
        k, p, x = int(args[0][0]), args[0][1], int(args[0][2])
        if p < 0 or p > 1 or k < 0 or x < k or k > 20000 or x > 20000:
            raise ValueError("You wrote a wrong input!")
        return(binCoeff(x - 1, k - 1)*(p**k)*((1 - p)**(x - k)))
    except ValueError as err:
        print("ERROR: ", err)
        exit()

def hipergeometric(*args):
    try:
        verify = lambda x: x > 20000 or x < 0
        k, n, Ntot, Ktot = int(args[0][0]), int(args[0][1]), int(args[0][2]), int(args[0][3])
        if all([verify(i) for i in [k, n, Ntot, Ktot]]):
            raise ValueError("Verify your input again mate!! >:(")
        num = binCoeff(Ktot, k) * binCoeff(Ntot - Ktot, n - k)
        den = binCoeff(Ntot, n)
        return(num/den)
    except ValueError as err:
        print("ERROR: ", err)
        exit()






if __name__ == '__main__':
    main()
