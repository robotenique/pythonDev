from matplotlib.animation import FuncAnimation
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
import subprocess as sbp

'''
    Rotates the color vector of the image into the 3D space,
    by the angle 'theta' provided, then generates a gif image.

    Probably only works with JPG, untested in other image formats.

    -> My guess is that this will be unreadeable >.<
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
    # Do matrix multiplication :O
    print(i)
    iR = np.einsum("ijk, lk->ijl", iN, rM(i * np.pi/50))
    im2 = uN(iR)
    ax.imshow(im2)
    ax.set_axis_off()
    plt.tight_layout()


def showImages():
    print("Showing images in the current directory (JPG only supported): ")
    sbp.Popen("ls -c | egrep \".\.jpg\"",shell=True)

showImages()
fig, ax = plt.subplots(dpi=100)
imgName = input("Type the name of image (for ex. img.jpg) and press [ENTER]:")
try:
    im = img.imread(imgName)
    fig.set_size_inches((im.shape[0]/100,im.shape[1]/100))
except Exception:
    print("Sorry, this file can't be opened! D:: Check the name again!")
    exit()
anim = FuncAnimation(fig, upd, frames=np.arange(0,100), interval=8)
anim.save("top.gif", dpi=100, writer="imagemagick")
plt.close()
print("GIF GENERATED! Open the file \"top.gif\" to see the effects!")
