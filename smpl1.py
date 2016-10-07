'''
1.10 - Control Structures
'''
def ex1():
    wordlist = ['cat','dog','rabbit']
    letterlist = [ ]
    for aword in wordlist:
        for aletter in aword:
            letterlist.append(aletter)
    #Using SET
    print(list(set(letterlist)))

def ex2():
    wordlist = ['cat','dog','rabbit']
    #List comprehension
    letterlist = [letter for word in wordlist for letter in word]
    print(list(set(letterlist)))

def main():
    ex1()
    ex2()

if __name__ == '__main__':
    main()
