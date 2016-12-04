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
    lenProb = 3
    menu = [[] for x in range(lenProb)]
    menu[0] = [
        "--- Welcome to the ProbDistributions =) ---\n",
        "You may choose one of the options below: \n",
        "[1] - Binomial Distribution\n",
        "[2] - Poisson Distribution\n"
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
    readIn = lambda x: float(input(x + " = "))
    probDistr = [0, binomial, poisson]
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



if __name__ == '__main__':
    main()
