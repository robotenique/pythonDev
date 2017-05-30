import pygame
import random
import math
import time

'''
    Simple BFS animation!
'''

SQRATIO = 4
NODERATIO = int(SQRATIO*1.5)
running = True
c = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
(width, height) = (1700, 800)
pygame.display.set_caption('BFS visualizer')
screen = pygame.display.set_mode((width, height))
cBall = lambda : (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
centerC = lambda p : (p.x + NODERATIO//2, p.y + NODERATIO//2)
n = 200

def isBorder(x, y):
    if x < 0 or x > width or y < 0 or y > height:
        return True
    return False

def displayWindow():
    global running
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

class Node():
    def __init__(self, x, y, depth):
        self.x = x
        self.y = y
        self.depth = depth
        self.steps = 0

    def move(self):
        self.y += SQRATIO
        self.steps += 1

    def isOld(self):
        return self.steps > 15

    def moveHorizontal(self, pos=False):
        if pos:
            self.x += SQRATIO
        else:
            self.x -= SQRATIO


nodeQueue = [Node(width//2, 20, 0)]
pygame.draw.circle(screen, cBall(), centerC(nodeQueue[0]), NODERATIO*3)
while running:
    if len(nodeQueue) > 0:
        p = nodeQueue[0]
        del nodeQueue[0]
        if  n//2**p.depth == n//2**(p.depth+1):
            displayWindow()
        p.move()
        while not p.isOld() and not isBorder(p.x, p.y):
            p.move()
            pygame.draw.rect(screen, c, (p.x, p.y, SQRATIO, SQRATIO))
            pygame.display.flip()
        v1, v2 = Node(p.x, p.y, p.depth+1), Node(p.x, p.y, p.depth+1)
        length = n//2**v1.depth
        for i in range(length):
            v1.moveHorizontal()
            v2.moveHorizontal(pos=True)
            pygame.draw.rect(screen, c, (v1.x, v1.y, SQRATIO, SQRATIO))
            pygame.draw.rect(screen, c, (v2.x, v2.y, SQRATIO, SQRATIO))
            pygame.display.flip()
        pygame.draw.circle(screen, cBall(), centerC(v1), NODERATIO)
        pygame.draw.circle(screen, cBall(), centerC(v2), NODERATIO)
        time.sleep(0.1)
        pygame.display.flip()
        nodeQueue.append(v1)
        nodeQueue.append(v2)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
