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
        self.PinB = None

    def setNextPin(self, source):
        if self.pinA == None:
            self.pinA = source
        else:
            if self.pinB == None:
                self.pinB = source
            else:
                raise RuntimeError("ERROR: NO EMPY PINS!")


    def getPinA(self):
        if self.pinA == None:
            return int(input("Enter pin A input for gate "+self.getName()+": "))
        else:
            self.pinA.getFrom().getOutput()


    def getPinB(self):
        if self.pinB == None:
            return int(input("Enter pin B input for gate "+self.getName()+": "))
        else:
            self.pinB.getFrom().getOutput()


class UnaryGate(LogicGate):
    """docstring for UnaryGate
        Implements a Logic Gate which receives one input
    """
    def __init__(self, name):
        LogicGate.__init__(self, name)
        self.pin = None


    def getPin(self):
        return int(input("Enter pin input for gate "+self.getName()+": "))


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




def main():
    gate = NotGate("topKEK")
    print(gate.getOutput())
if __name__ == '__main__':
    main()
