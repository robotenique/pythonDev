import string
import random as rnd
'''
1.12 - Defining FUnctions
--> The infinite monkey theorem <--
- sentence: "methinks it is like a weasel"
'''


# Return a random string, provided a placeholder
# The placeholder tells which position of the generated string will me random
def rndString(n, plc, target):
    xStr = list()
    for i in range(n):
        if not plc[i]:
            xStr.append(rnd.choice(string.ascii_lowercase+" "))
        else:
            xStr.append(target[i])
    return "".join(xStr)


def score(strGen, target, plc, n):
    print(strGen)
    score = [strGen[i] == target[i] for i in range(n)]
    if sum(score) > sum(plc):
        plc = score
    return sum(plc), plc


def checker(target, n, plc):
    maxN = counter = 0
    bstStr = ""
    while maxN != n:
        strGen = rndString(n, plc, target)
        nGen, plc = score(strGen, target, plc, n)
        if nGen > maxN:
            maxN = nGen
            bstStr = strGen
        counter += 1
        if counter % 1000 == 0:
            print("Best String: %s , Score = %d" % (bstStr, maxN))
            print("PLC = %s " % (plc))
    print("Success! String found in %d tries!" % (counter - 1))


def main():
    targetString = "methinks it is like a weasel"
    n = len(targetString)
    plc = [0]*n
    checker(targetString, n, plc)

if __name__ == '__main__':
    main()
