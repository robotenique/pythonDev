import matplotlib.pyplot as plt
import matplotlib.image as img
from matplotlib.animation import FuncAnimation
import numpy as np


def main():
    im = img.imread("oi.jpg")
    print(im.shape)
    rotateColor(im)


def undo_normalise(im):
    return (1 + 1/(np.exp(-im) + 1) * 257).astype("uint8")

def normalize(im):
	return -np.log(1/((1 + im)/257)- 1)

def rotationMatrix(theta):
	'''
	3D rotation of matrix by the angle 'theta'
	'''
	return np.c_[
		[1,0,0],
		[0, np.cos(theta), -np.sin(theta)],
		[0, np.sin(theta), np.cos(theta)]]

def plotImage(im, h = 8, **kwargs):
	# Slice Image : im = im[100:300,:200,:] 
	y = im.shape[0]
	x = im.shape[1]
	w = (y/x) * h	
	plt.figure(figsize = (w,h))
	plt.imshow(im, interpolation = "none", **kwargs)
	plt.axis('off')
	plt.show()
	'''	
	fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
	for c, ax in zip(range(3), axs):
		# Convert data types to INT
		tmp = np.zeros(im.shape, dtype="uint8")
		tmp[:, : ,c] = im[: , :, c]
		ax.imshow(tmp)
		ax.set_axis_off()		
	plt.show()
	'''

def rotateColor(im):
	im_Norm = normalize(im)
	im_Rotated = np.einsum("ijk,lk->ijl", im_Norm, rotationMatrix(np.pi))
	im2 = undo_normalise(im_Rotated)
	plotImage(im2)



if __name__ == '__main__':
    main()
