import pygame
import random
import math
import time
(width, height) = (400, 400)
pygame.display.set_caption('Random path')
screen = pygame.display.set_mode((width, height))


r = random.randint(200, 255)
g = random.randint(0, 255)
b = random.randint(0, 255)
x = random.randint(0, 10)
y = random.randint(0, height - 1)
sqRatio = 2
cRatio = 36
running = True
isSame = False
struct = {}
for k in range(width//2 - cRatio, width//2 + cRatio + 1):
    struct[(k, math.sqrt(cRatio**2 - (k - width//2)**2)+height//2)] = True
    struct[(k, -math.sqrt(cRatio**2 - (k - width//2)**2)+height//2)] = True


def isBorder(x, y):
    if x < 0 or x > width or y < 0 or y > height:
        return True
    return False


def getPartcl():
    global x, y, r, g, b, isSame
    if isSame:
        if isBorder(x, y):
            isSame = False
        else:
            rnd = random.randint(0, 3)
            if rnd == 0:
                x += sqRatio
            elif rnd == 1:
                y += sqRatio
            elif rnd == 2:
                x -= sqRatio
            else:
                y -= sqRatio
    else:
        if random.random() < 0.5:
            x = random.randint(0, width - 1)
            if random.random() < 0.5 :
                y = random.randint(height - 10, height - 1)
            else:
                y = random.randint(0, 11)
        else:
            y = random.randint(0, height - 1)
            if random.random() < 0.5:
                x = random.randint(width - 10, width - 1)
            else:
                x = random.randint(0, 11)
        isSame = True

inPlace =  lambda x, y : (x - width//2)**2 + (y - height//2)**2

def inStructure(x, y):
    neigh = [(x + sqRatio, y),
             (x , y + sqRatio),
             (x - sqRatio, y),
             (x, y - sqRatio)]
    return any(x in struct for x in neigh)


def getColor(t):
    return (200 - (t%255)/5, random.randint(150, 255), (50+t%255)%255)

for k in struct:
    pygame.draw.rect(screen, (r, g, b), (k[0], k[1], sqRatio, sqRatio))

t = 0
while running:
    while not inStructure(x, y):
        getPartcl()
    t += 1
    struct[(x, y)] = True
    pygame.draw.circle(screen, getColor(t),(x,y),sqRatio,0)
    isSame = False
    getPartcl()
    pygame.display.flip()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
