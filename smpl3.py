from random import randint as rndInt

def main():
    l = list([rndInt(1,100) for x in range(rndInt(5,900))])
    # min_1(l)
    # min_2(l)

# Simple O(n²)
def min_1(list):
    minOut = list[0]
    for i in list:
        minLoc = i
        for j in list:
            if j < minLoc:
                minLoc = j
        if minLoc < minOut:
            minOut = minLoc
    print(minOut)

# Typical O(n) approach
def min_2(list):
    minL = list[0]
    for i in list:
        if i < minL:
            minL = i
    print(minL)

if __name__ == '__main__':
    main()
