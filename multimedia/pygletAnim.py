import pyglet
import time
import random as rnd
import itertools as itools
from abc import ABC, abstractmethod

class Particle(ABC):
    UNIT_CONV = None
    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def check_collision(self):
        pass

    @abstractmethod
    def build_movement(self):
        pass

    def set_pos(self, x, y):
        if not Particle.UNIT_CONV:
            raise AttributeError("You must set a base unit conversion!")
        self.x = x
        self.y = y



class BallMRUV(Particle):
    def __init__(self, **kwargs):
        super(Particle, set_pos(kwargs['x'], kwargs['y']))
        self.build_movement(**kwargs)

    def step(self):
        dt = self.deltaT
        self.vx += self.ax*dt
        self.vy += self.ay*dt
        self.x  += self.vx*dt
        self.y  += self.vy*delta

    def check_collision(self):
        pass

    def build_movement(**kwargs):
        convert = lambda i : i*super(self.__class__).UNIT_CONV
        self.vx = convert(kwargs['vx'])
        self.vy = convert(kwargs['vy'])
        self.ax = convert(kwargs['ax'])
        self.ay = convert(kwargs['ay'])
        self.deltaT = kwargs['deltaT']








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
        self.x = kwargs['x0']
        self.y = kwargs['y0']
        def genF():
            acX = kwargs['acX']
            acY = kwargs['acY']

            x = kwargs['x']
            vx = kwargs['vx']
            x0 = kwargs['x0']

            y = kwargs['y']
            vy = kwargs['vy']
            y0 = kwargs['y0']
            self.kwargs['ax'] = 0 if not kwargs.get('ax') else kwargs['ax']
            self.kwargs['ay'] = 0 if not kwargs.get('ay') else kwargs['ay']


            deltaT = kwargs['deltaT']
            k = kwargs['k']
            for i in itools.count():
                if acX:
                    vx -= k*x*deltaT
                    x += vx*deltaT
                if acY:
                    vy -= k*y*deltaT
                    y += vy*deltaT
                self.kwargs['vx'] = vx
                self.kwargs['vy'] = vy
                self.x = x + x0
                self.y = y + y0
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
                self.kwargs['vx'] = vx
                self.kwargs['vy'] = vy
                yield x, y
        return genF

    def startPos(self):
        return self.x, self.y

    def reverse(self, x=False, y=False, ax=False, ay=False):
        supportedMovs = ('MRU', 'MRUV', 'SHO')
        if self.movType not in supportedMovs:
            raise NotImplementedError("You can't reverse this velocity!")
        if x:
            self.kwargs['vx'] = -self.kwargs['vx']
        if y:
            self.kwargs['vy'] = -self.kwargs['vy']
        if ax and self.movType != 'SHO':
            self.kwargs['ax'] = -self.kwargs['ax']
        if ay and self.movType != 'SHO':
            self.kwargs['ay'] = -self.kwargs['ay']
        self.kwargs['x'] = self.x
        self.kwargs['y'] = self.y
        return self.createMRUV(**self.kwargs)()

class BallAnimation(pyglet.window.Window):
    def __init__(self, texture=None, movs=None, *args, **kwargs):
        super(BallAnimation, self).__init__(*args, **kwargs)
        self.width = kwargs['width']
        self.height = kwargs['height']
        self.batch = pyglet.graphics.Batch()
        self.spriteList = []
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
        pass
        '''w = self.sprite.width * 3
        h = self.sprite.height * 3
        self.width = w
        self.height = h'''

    def moveObjects(self, t):
        for spriteItem in self.spriteList:
            sprite = spriteItem[0]
            step = spriteItem[2]
            self.checkBoundaries(spriteItem)
            sprite.x, sprite.y = next(spriteItem[2])
            sprite.rotation += 5

    def checkBoundaries(self, spriteItem):
        sprite = spriteItem[0]
        mov = spriteItem[3]
        revX, revY = True, True
        if sprite.x > self.width - sprite.width/2:
            sprite.x = self.width - sprite.width/2
        elif sprite.x < sprite.width/2:
            sprite.x = sprite.width/2
        else:
            revX = False
        if revX:
            mov.x = sprite.x
            spriteItem[2]  = mov.reverse(x=True, ax=True)

        if sprite.y > self.height - sprite.height/2:
            sprite.y = self.height - sprite.height/2
        elif sprite.y < sprite.height/2:
            sprite.y = sprite.height/2
        else:
            revY = False
        if revY:
            mov.y = sprite.y
            spriteItem[2] = mov.reverse(y=True, ay=True)

    def on_draw(self):
        self.clear()
        self.batch.draw()


def main():
    deltaT = 1/60
    txt = ['redBall.png','greenBall.png', 'purpleBall.png']
    redBall = BallMRUV(x=200, y=500, vx=10, vy=20, deltaT=deltaT)
    movs = [Movement(x=200, y=500, vx=992, vy=220, deltaT=deltaT),
            Movement(movType='SHO', x=100, y=200, vx=0, vy=20,
                     acX=True, acY=True, x0=683, y0=500, deltaT=deltaT, k=10),
            Movement(movType='MRUV', x=200,vx=10,ax=200,y=200,vy=0,ay=-200 ,deltaT=deltaT)]

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
