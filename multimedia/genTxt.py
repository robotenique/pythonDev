# python >= 3.6
import sys
import numpy as np
from PIL import Image as im
from PIL import ImageEnhance as imEnh
'''
Generate output with the specific x,y coordinates where the
image is black.
'''
def usage():
    print("Generate output with the specific x,y coordinates where the "
      "image is black. \nUsage: ./genTxt.py <img> <contrast> > draw.txt"
       "\ne.g.: ./genTxt.py topper.png 2.5 > in.txt"
       "img - png or jpg \n contrast (float) -  0 (minimum)...1(no changes)...2...3, etc")
    exit()

sz = (100,100)

if(len(sys.argv) < 2):
    usage()

img = im.open(sys.argv[1])
if(img.size[0] > 120 or img.size[1] > 120 or img.size[0] != img.size[1]):
    img.thumbnail(sz, im.ANTIALIAS)
    sz = (img.size[0] , img.size[1])

img = imEnh.Contrast(img).enhance(float(sys.argv[2])).convert('1')
bkg = im.new('RGBA', sz, (255, 255, 255, 0))
bkg.paste(img, (0, 0))
bkg.show()

tarr = np.asarray(bkg.convert('1'))
points = list(f"{tarr.shape[0]}\n")
for i in range(tarr.shape[0]):
    for j in range(tarr.shape[1]):
        if tarr[i][j]:
            points.append( f"{i} {j}\n")

for pixel in points:
    print(pixel, end="")
