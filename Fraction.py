# gdc(a,b) = gcd(b, a%b) OBS: Positive integer only
def gcd(m, n):
    while(m % n != 0):
        aux = m
        aux2 = n
        m = aux2
        n = aux % aux2
    return n


class Fraction(object):
    """ Represents a fraction, which consists of two integer numbers:
        The first one is called the numerator, and the second one is
        the denominator.

        To instantiate an object of this class you need to provide a
        default numerator and denominator to the init method.
        """
    def __init__(self, num, den):
        if num is not int or den is not int:
            raise RuntimeError("Only integers values supported for fraction!")
        if den == 0:
            raise RuntimeError("ERROR: denominator equals zero!")
        common = gcd(num, den)
        self.num = num//common
        self.den = den//common

    # Overriding string conversion method
    def __str__(self):
        return str(self.num) + " / " + str(self.den)

    # Overriding the arithmetic operation for Fractions
    def __add__(self, other):
        den = other.den * self.den
        num = (self.num*den//self.den)+(other.num*den//other.den)
        return Fraction(num, den)

    def __sub__(self, other):
        other.num *= -1
        return self.__add__(other)

    def __mul__(self, other):
        num = self.num * other.num
        den = self.den * other.den
        return Fraction(num, den)

    def __div__(self, other):
        other.num, other.den = other.den, other.num
        if other.den == 0:
            raise RuntimeError("ERROR: denominator equals zero!")
        return self.__mul__(other)

    # Right way to check if two fractions are equal
    def __eq__(self, other):
        # Cross product
        num1 = self.num*other.den
        num2 = self.den*other.num
        return num1 == num2

    def __lt__(self, other):
        return(self.num/self.den < other.num/other.den)

    def __gt__(self, other):
        return(other.__lt__(self))

    def getNum(self):
        return self.num

    def getDen(self):
        return self.den

    def __truediv__(self, other):
        divA = self.num / self.den
        divB = other.num / other.den
        return divA / divB


def main():
    myf = Fraction(1, 22)
    myf2 = Fraction(1, 2)
    print(myf != myf2)




if __name__ == '__main__':
    main()
