from matplotlib.animation import FuncAnimation
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np

'''
    Rotates the color vector of the image into the 3D space,
    by the angle 'theta' provided, then generates a gif image.
'''

def n(im):
    '''
    Normalize the image
    '''
    return -np.log(1/((1 + im)/257) - 1)

def uN(im):
    '''
    Un-Normalize the image
    '''
    return (1 + 1/(np.exp(-im) + 1) * 257).astype("uint8")

def rM(t):
    '''
    Rotation Matrix in 3D by the angle 't'
    '''
    return np.c_[
    [1,0,0],
    [0,np.cos(t), -np.sin(t)],
    [0,np.sin(t), np.cos(t)]
    ]

def upd(i):
    iN = n(im)
    # Do matrix multiplication
    iR = np.einsum("ijk, lk->ijl", iN, rM(i * np.pi/10))
    im2 = uN(iR)
    ax.imshow(im2)
    ax.set_title("BEM LOCO")
    ax.set_axis_off()

fig, ax = plt.subplots(figsize=(10,10))
im = img.imread("oi.jpg")
anim = FuncAnimation(fig, upd, frames=np.arange(0,50), interval=50)
anim.save("top.gif", dpi=80, writer="imagemagick")
plt.close()
