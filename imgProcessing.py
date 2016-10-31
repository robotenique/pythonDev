import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np


def main():
    im = img.imread("oi.jpg")
    print(im.shape)
    plotImage(im)

def plotImage(im, h = 8, **kwargs):
	# Slice Image : im = im[100:300,:200,:] 
	y = im.shape[0]
	x = im.shape[1]
	w = (y/x) * h
	''' 
	plt.figure(figsize = (w,h))
	plt.imshow(im, interpolation = "none", **kwargs)
	plt.axis('off')
	plt.show()
	''' 
	fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15,5))
	for c, ax in zip(range(3), axs):
		# Convert data types to INT
		tmp = np.zeros(im.shape, dtype="uint8")
		tmp[:,:,c] = im[: , :, c]
		ax.imshow(tmp)
		ax.set_axis_off()
	
	plt.show()





if __name__ == '__main__':
    main()
