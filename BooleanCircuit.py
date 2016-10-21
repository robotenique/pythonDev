# Implements Polymorphism and Inheritance in O.O.
class LogicGate:
    """docstring for LogicGate
        Implements a logic gate superclass.
    """
    def __init__(self, name):
        self.name = name
        self.output = None


    def getName(self):
        return self.name


    def getOutput(self):
        # Noice Polymorphism O.O.
        # Noice Polymorphism O.O
        self.output = self.performGateLogic()
        return self.output

class BinaryGate(LogicGate):
    """docstring for BinaryGate
        Implements a Logic Gate which receives two inputs
    """
    def __init__(self, name):
        LogicGate.__init__(self, name)
        self.pinA = None
        self.pinB = None

    def setNextPin(self, source):
        if self.pinA == None:
            self.pinA = source
        else:
            if self.pinB == None:
                self.pinB = source
            else:
                raise RuntimeError("ERROR: NO EMPTY PINS!")


    def getPinA(self):
        if self.pinA == None:
            return int(input("Enter pin A input for gate "+self.getName()+": "))
        else:
            return self.pinA.getFrom().getOutput()


    def getPinB(self):
        if self.pinB == None:
            return int(input("Enter pin B input for gate "+self.getName()+": "))
        else:
            return self.pinB.getFrom().getOutput()

class UnaryGate(LogicGate):
    """docstring for UnaryGate
        Implements a Logic Gate which receives one input
    """
    def __init__(self, name):
        LogicGate.__init__(self, name)
        self.pin = None


    def setNextPin(self, source):
        if self.pin == None:
            self.pin = source
        else:
            print(" CANNOT CONNECT: NO EMPTY PIN!")

    def getPin(self):
        if self.pin == None:
            return int(input("Enter pin input for gate "+self.getName()+": "))
        else:
            return self.pin.getFrom().getOutput()

class AndGate(BinaryGate):
    """docstring for AndGate
    Implements a 'AND' logic gate
    """
    def __init__(self, n):
        BinaryGate.__init__(self, n)


    def performGateLogic(self):
        a = self.getPinA()
        b = self.getPinB()
        if a == 1 and b == 1:
            return 1
        return 0

class OrGate(BinaryGate):
    """docstring for OrGate
        Implements a 'OR' logic gate
    """
    def __init__(self, name):
        BinaryGate.__init__(self, name)


    def performGateLogic(self):
        a = self.getPinA()
        b = self.getPinB()
        if a == 1 or b == 1:
            return 1
        return 0

class NotGate(UnaryGate):
    """docstring for NotGate
        Implements the "NOT" unary gate
    """
    def __init__(self, name):
        UnaryGate.__init__(self, name)


    def performGateLogic(self):
        pin = self.getPin()
        if pin == 1:
            return 0
        return 1

class Conector:
    """docstring for Conector
        Connects two gates together, redirecting the input and output
    """
    def __init__(self, fgate, tgate):
        self.fromGate = fgate
        self.toGate = tgate
        tgate.setNextPin(self)


    def getFrom(self):
        return self.fromGate


    def getTo(self):
        return self.toGate

class NandGate(BinaryGate):
    """docstring for NandGate
        Implements a NAND Gate
    """
    def __init__(self, name):
        BinaryGate.__init__(self, name)

    def performGateLogic(self):
        a = self.getPinA()
        b = self.getPinB()
        if a == 1 and b == 1:
            return 0
        return 1

class NorGate(BinaryGate):
    """docstring for NorGate
        Implements a NOR gate!
    """
    def __init__(self, name):
        BinaryGate.__init__(self, name)

    def performGateLogic(self):
        a = self.getPinA()
        b = self.getPinB()
        if a == 1 or b == 1:
            return 0
        return 1


def main():
    # Here we proof that :
    # NOT ((A and B) or (C and D)) equals NOT(A and B) and NOT(C and D)

    # Equivalence 1
    g1 = AndGate("G1")
    g2 = AndGate("G2")
    g3 = NorGate("G3")
    c1 = Conector(g1, g3)
    c2 = Conector(g2, g3)

    #Equivalence 2
    g_1 = NandGate("g_1")
    g_2 = NandGate("g_2")
    g_3 = AndGate("g_3")
    c_1 = Conector(g_1, g_3)
    c_2 = Conector(g_2, g_3)

    print(g3.getOutput())
    print(g_3.getOutput())
if __name__ == '__main__':
    main()
