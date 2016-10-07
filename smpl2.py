import string
import random as rnd
'''
1.12 - Defining FUnctions
--> The infinite monkey theorem <--
- sentence: "methinks it is like a weasel"
'''
# Return a random string
def rndString(n):
    return "".join([rnd.choice(string.ascii_lowercase+" ") for i in range(n)])

def score(strGen, target):
    score = [strGen[i]==target[i] for i in range(min(len(strGen), len(target)))]
    return sum(score)

def checker(target, n):
    maxN = counter = 0
    bstStr = ""
    while maxN != n:
        strGen = rndString(n)
        nGen = score(strGen, target)
        if nGen > maxN:
            maxN = nGen
            bstStr = strGen
        counter += 1
        if counter%1000 == 0:
            print("Best String: %s , Score = %d" %(bstStr, maxN))
    print("Success! String found in %d tries!" %(counter - 1))


def main():
    targetString = "methinks it is like a weasel"
    n = len(targetString)
    checker(targetString, n)

if __name__ == '__main__':
    main()
