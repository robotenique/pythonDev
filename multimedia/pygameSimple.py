import pygame
import itertools as it
from time import sleep

SCREEN_WIDTH = 1366
SCREEN_HEIGHT = 728

BKG_COLOR = (0,0,0)
CORNER_COLOR = (142, 112, 119)
FPS = 30
pygame.init()

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Simulation 1")
myfont = pygame.font.SysFont("monotype", 45, bold = True)
tFont = pygame.font.SysFont("liberation", 100)
finalFont = pygame.font.SysFont("liberation", 100)
label = myfont.render("R0b0tenique", 1, (33, 224, 195))
finalL = finalFont.render("YOU LOSE!", 1, (150, 20, 0))
timeL = tFont.render("30", 1, (145, 255, 0))
wF, hF = finalL.get_width(), finalL.get_height()
w, h = label.get_width(), label.get_height()
wT, hT = timeL.get_width, timeL.get_height
initX, initY = SCREEN_WIDTH/2 - w, SCREEN_HEIGHT/2 - h
screen.blit(label, (initX, initY))

# - objects -
rectangle = pygame.rect.Rect((initX, initY), (w, h))
#alias
r = pygame.rect.Rect
corners = list()
cornerCords =  [[0, 0], [SCREEN_WIDTH - (w + 10), 0], [0, SCREEN_HEIGHT - (h + 30)],
                [SCREEN_WIDTH - (w + 10), SCREEN_HEIGHT - (h + 30)]]

for c in cornerCords:
    corners.append(r((c[0], c[1]), (w + 10, h + 30)))
rectangle_draging = False

# image
img = pygame.image.load("player.gif")
imgW, imgH = img.get_size()
# - mainloop -
clock = pygame.time.Clock()
running = True

timer = pygame.time.get_ticks() #starter tick
while running:
    # - events -
    dt = (pygame.time.get_ticks()-timer)/1000 #passed time
    if dt > 10:
        screen.fill(BKG_COLOR)
        screen.blit(finalL, (SCREEN_WIDTH/2 - wF/2, SCREEN_HEIGHT/2 - hF/2))
        pygame.display.flip()
        sleep(3)
        break
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                if rectangle.collidepoint(event.pos):
                    rectangle_draging = True
                    mouse_x, mouse_y = event.pos
                    offset_x = rectangle.x - mouse_x
                    offset_y = rectangle.y - mouse_y

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                rectangle_draging = False

        elif event.type == pygame.MOUSEMOTION:
            if rectangle_draging:
                mouse_x, mouse_y = event.pos
                rectangle.x = mouse_x + offset_x
                rectangle.y = mouse_y + offset_y

    # - updates (without draws) -

    # empty

    # - draws (without updates) -

    screen.fill(BKG_COLOR)
    timeL = tFont.render(str(round(dt)), 1, (145, 255, 0))

    screen.blit(img, (SCREEN_WIDTH/2, SCREEN_HEIGHT/2 - imgH/2))
    for c in corners:
        pygame.draw.rect(screen, CORNER_COLOR, c)
    screen.blit(label, (rectangle.x, rectangle.y))
    pygame.draw.rect(screen, (0,123, 243), rectangle, 1)
    screen.blit(timeL, (SCREEN_WIDTH/2 - wT()/2, 10))
    pygame.display.flip()

    # - constant game speed / FPS -

    clock.tick(FPS)

# - end -

pygame.quit()
