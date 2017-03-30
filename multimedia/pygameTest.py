import random as rnd
import pygame
import sys

def generateObj():
    objPos = (rnd.randint(50, 950), rnd.randint(50, 950))
    objColor = (0, 0, 0)
    return list([objColor, objPos])


pygame.init()
bgcolor = (255, 255, 204)
surf = pygame.display.set_mode((1000,1000))

circleColor = (255, 51, 51)
x, y = 500, 500
circleRad = 50
objRad = 25


pygame.display.set_caption("TOOOPPPER!")
obj = generateObj()
change = False

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            bgcolor = (rnd.randint(50,255), rnd.randint(200,255), rnd.randint(200,255))
            circleColor = (rnd.randint(0,255), rnd.randint(0,255), rnd.randint(0,255))
            if(change == True):
                obj = generateObj()
                change = False

            if event.key == pygame.K_UP:
                y -= 40
            elif event.key == pygame.K_DOWN:
                y += 40
            elif event.key == pygame.K_RIGHT:
                x += 40
            elif event.key == pygame.K_LEFT:
                x -= 40


    circlePos = (x % 1000, y % 1000)
    surf.fill(bgcolor)
    if((circlePos[0] - obj[1][0])**2 + (circlePos[1] - obj[1][1])**2 <= (objRad+circleRad)**2):
        obj[1] = (-400, -400)
        circleRad += 20
        change = True
    if(circleRad >= 450):
        sys.exit()
    pygame.draw.circle(surf, circleColor, circlePos, circleRad)
    pygame.draw.circle(surf, obj[0], obj[1], objRad)
    pygame.display.flip()
