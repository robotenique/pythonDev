import turtle
import random
from time import sleep

def rec_spiral(d, depth):
    #Koch star
    sz = d
    if depth == depthLim:
        spiral.fd(sz)
        return
    rec_spiral(d/3, depth + 1)
    spiral.lt(60)
    rec_spiral(d/3, depth + 1)
    spiral.rt(120)
    rec_spiral(d/3, depth + 1)
    spiral.lt(60)
    rec_spiral(d/3, depth + 1)
    return

d = 500.0       #size of the fractal size
depthLim = 6    #recursion limit
turtle.bgcolor("black")
turtle.setup( width = 800, height = 800)
spiral = turtle.Turtle()
spiral.speed(0)



'''
#Random figures!

for i in range(300):
    spiral.pencolor(random.random(), random.random(), random.random())
    spiral.fd(i+10)
    spiral.rt(random.randint(20, 360))
    verifyPos
'''

#spiral.pen(pensize=2)
spiral.pencolor(random.random(), random.random(), random.random())
spiral.hideturtle()           #make the turtle invisible
spiral.penup()                #don't draw when turtle moves
spiral.goto(spiral.pos()[0], spiral.pos()[1] - 200)       #move the turtle to a location
spiral.showturtle()           #make the turtle visible
spiral.pendown()              #draw when the turtle moves
spiral.lt(120)
for i in range(30):
    k = 0
    spiral.pencolor(random.random(), random.random(), random.random())
    while(k < 3):
        rec_spiral(d, 0)
        spiral.rt(120)
        k +=1
    spiral.bk(5)
'''
    rec_spiral(d, 0)
    d += 30
    spiral.fd(15)
    spiral.rt(90)
    spiral.fd(2)
    spiral.rt(90)
    '''
turtle.done()
