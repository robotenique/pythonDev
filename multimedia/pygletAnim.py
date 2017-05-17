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
        elif(movType == 'MRUV'):
            self.movGenerator = self.createMRUV(**kwargs)
        else:
            raise NotImplementedError("This movement isn't implemented!")

    def createSHO(self, **kwargs):
        """
        Creates a SHO movement with the given parameters and returns a
        generator of the movement.

        Parameters
        ----------
        acX, acY: boolean
            Describes whether to apply the SHO movement to x or y coordinates
        x0, y0: int (pixel unit)
            The equilibrium state position of both x and y coordinates
        x, y : int (pixel units)
            The initial distortion of the oscillator from the equilibrium state
        vx, vy : float
            The initial velocity of the x and y coordinates

        Returns
        -------
        Generator of movement steps which in turns yield the x and y coords.
        """
        self.x = kwargs['x']
        self.y = kwargs['y']
        def genF():
            acX = kwargs['acX']
            acY = kwargs['acY']

            x = kwargs['x']
            vx = kwargs['vx']
            x0 = kwargs['x0']

            y = kwargs['y']
            vy = kwargs['vy']
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
                yield x + x0, y + y0
        return genF

    def createMRU(self, **kwargs):
        self.kwargs['ax'], self.kwargs['ay'] = 0, 0
        return self.createMRUV(**self.kwargs)

    def createMRUV(self, **kwargs):
        self.x = kwargs['x']
        self.y = kwargs['y']
        def genF():
            x = kwargs['x']
            vx = kwargs['vx']
            ax = kwargs['ax']
            y = kwargs['y']
            vy = kwargs['vy']
            ay = kwargs['ay']
            deltaT = kwargs['deltaT']
            for c in itools.count():
                vx += ax*deltaT
                vy += ay*deltaT
                x += vx*deltaT
                y += vy*deltaT
                self.x = x
                self.y = y
                yield x, y
        return genF

    def startPos(self):
        return self.x, self.y

    def reverseVelocity(self, x=True, y=True):
        supportedMovs = ('MRU', 'MRUV')
        if self.movType not in supportedMovs:
            raise NotImplementedError("You can't reverse this velocity!")
        if x:
            self.kwargs['vx'] = -self.kwargs['vx']
        if y:
            self.kwargs['vy'] = -self.kwargs['vy']
        self.kwargs['x'] = self.x
        self.kwargs['y'] = self.y
        return self.createMRUV(**self.kwargs)()

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
        spx, spy = mov.startPos()
        sprite.position = (spx, spy)
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
            sprite.rotation += 5
            self.checkBoundaries(spriteItem)

    def checkBoundaries(self, spriteItem):
        # TODO : Encapsulates the collision verification right here
        sprite = spriteItem[0]
        mov = spriteItem[3]
        if sprite.x > self.width or sprite.x < 0:
            sprite.x = 100 if sprite.x < 0 else self.width - 100
            spriteItem[2]  = mov.reverseVelocity(y=False)
        if sprite.y > self.height or sprite.y < 0:
            sprite.y = 100 if sprite.y < 0 else self.height - 100
            spriteItem[2] = mov.reverseVelocity(x=False)


    def on_draw(self):
        self.clear()
        self.batch.draw()

def main():
    deltaT = 1/60
    txt = ['redBall.png','greenBall.png', 'purpleBall.png']
    movs = [Movement(x=200, y=500, vx=800, vy=400, deltaT=deltaT),
            Movement(movType='SHO', x=100, y=200, vx=0, vy=0,
                     acX=True, acY=True, x0=683, y0=500, deltaT=deltaT, k=10),
            Movement(movType='MRUV', x=200,vx=10,ax=200,y=500,vy=300,ay=-9.8,deltaT=deltaT)]

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
