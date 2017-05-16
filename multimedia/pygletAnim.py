import pyglet
import time
import random as rnd
import itertools as itools

class Movement:
    def __init__(self, movType='MRU', **kwargs):
        self.kwargs = kwargs
        self.movType = movType
        if(movType == 'MRU'):
            self.movGenerator = self.createMRU(**kwargs)
        elif(movType == 'SHO'):
            self.movGenerator = self.createSHO(**kwargs)
        else:
            raise NotImplementedError("This movement isn't implemented!")


    def createSHO(self, **kwargs):
        self.x = kwargs['x']
        self.y = kwargs['y']
        def genF():
            x = kwargs['x']
            vx = kwargs['vx']
            acX = kwargs['acX']
            x0 = kwargs['x0']

            y = kwargs['y']
            vy = kwargs['vy']
            acY = kwargs['acY']
            y0 = kwargs['y0']

            deltaT = kwargs['deltaT']
            k = kwargs['k']
            for i in itools.count():
                if acX:
                    vx -= k*x*deltaT
                    x += vx*deltaT
                if acY:
                    vy -= k*y*deltaT
                    y += vy*deltaT
                self.x = x
                self.y = y
                yield x + x0, y
        return genF

    def createMRU(self, **kwargs):
        self.x = kwargs['x']
        self.y = kwargs['y']
        def genF():
            x = kwargs['x']
            y = kwargs['y']
            vx = kwargs['vx']
            vy = kwargs['vy']
            deltaT = kwargs['deltaT']
            for c in itools.count():
                x += vx*deltaT
                y += vy*deltaT
                self.x = x
                self.y = y
                yield x, y
        return genF

    def startPos(self):
        return self.x, self.y

    def reverseVelocity(self):
        if self.movType != 'MRU':
            raise NotImplementedError("You can't reverse this velocity!")
        self.kwargs['vx'] = -self.kwargs['vx']
        self.kwargs['vy'] = -self.kwargs['vy']
        self.kwargs['x'] = self.x
        self.kwargs['y'] = self.y
        return self.createMRU(**self.kwargs)()

class BallAnimation(pyglet.window.Window):
    def __init__(self, texture=None, movs=None, *args, **kwargs):
        super(BallAnimation, self).__init__(*args, **kwargs)
        self.batch = pyglet.graphics.Batch()
        self.spriteList = []
        self.width = 1366
        self.height = 768
        for t, mov in zip(texture, movs):
            img = pyglet.resource.image(t)
            self.createDrawableObjects(img, mov)
        self.adjustWindowSize()

    def createDrawableObjects(self, image, mov):
        image.anchor_x = image.width/2
        image.anchor_y = image.height/2
        sprite = pyglet.sprite.Sprite(image, batch=self.batch)
        sprite.position = (
            rnd.randint(int(sprite.width/2), self.width),
            rnd.randint(int(sprite.height/2), self.height))
        self.spriteList.append([sprite, False, mov.movGenerator(), mov])


    def adjustWindowSize(self):
        self.width = 1366
        self.height = 768
        '''w = self.sprite.width * 3
        h = self.sprite.height * 3
        self.width = w
        self.height = h'''


    def moveObjects(self, t):
        for spriteItem in self.spriteList:
            sprite = spriteItem[0]
            step = spriteItem[2]
            sprite.x, sprite.y = next(step)
            self.checkBoundaries(spriteItem)

    def checkBoundaries(self, spriteItem):
        return 1
        # TODO : Encapsulates the collision verification right here
        sprite = spriteItem[0]
        if sprite.x > self.width or sprite.x < 0:
            sprite.x = sprite.x + 100 if sprite.x < 0 else sprite.x - 100
            spriteItem[2]  = spriteItem[3].reverseVelocity()


    def on_draw(self):
        self.clear()
        self.batch.draw()

def main():
    deltaT = 1/60
    txt = ['redBall.png','greenBall.png']
    movs = [Movement(x=200, y=500, vx=500, vy=0, deltaT=deltaT),
            Movement(movType='SHO', x=10, y=500, vx=220, vy=110,
                     acX=True, acY=False, x0=683, y0=500, deltaT=deltaT, k=1)]

    '''
    # Creating a movement, running it 10 times, then reversing it
    # the running 10 times again!
    mv = Movement(x=600, y=500, vx=30, vy=0, deltaT=deltaT)

    mgen = mv.movGenerator()

    for i in range(10):
        print(next(mgen))

    mgen = mv.reverseVelocity()

    for i in range(10):
        print(next(mgen))

    '''
    window = BallAnimation(texture=txt, movs=movs, width=1366,
            height=768, caption='Pyglet ball animation')
    pyglet.gl.glClearColor(0,0, 0, 1) # Black background
    pyglet.clock.schedule_interval(window.moveObjects, deltaT)
    pyglet.app.run()

if __name__ == '__main__':
    main()
