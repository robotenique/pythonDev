import pygame

SCREEN_WIDTH = 1366
SCREEN_HEIGHT = 728

BKG_COLOR = (0,0,0)

FPS = 30

pygame.init()

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
myfont = pygame.font.SysFont("monotype", 45)
pygame.display.set_caption("Simulation 1")
# - objects -
# render text
label = myfont.render("DR. Eggman!", 1, (0, 200, 250))
screen.blit(label, (0, 0))
# render rect
lRect = pygame.rect.Rect((0, 0), (label.get_width(), label.get_height()))
print(lRect)
lRectDrag = False
# - mainloop -
clock = pygame.time.Clock()
running = True
while running:
    # - events -
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                if lRect.collidepoint(event.pos):
                    lRectDrag = True
                    mouse_x, mouse_y = event.pos
                    offset_x = lRect.x - mouse_x
                    offset_y = lRect.y - mouse_y

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                lRectDrag = False

        elif event.type == pygame.MOUSEMOTION:
            if lRectDrag:
                mouse_x, mouse_y = event.pos
                lRect.x = mouse_x + offset_x
                lRect.y = mouse_y + offset_y

    # - updates (without draws) -

    # empty

    # - draws (without updates) -

    screen.fill(BKG_COLOR)

    pygame.draw.rect(screen, BKG_COLOR, lRect)

    pygame.display.flip()

    # - constant game speed / FPS -

    clock.tick(FPS)

# - end -

#pygame.quit()
