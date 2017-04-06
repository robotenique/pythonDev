from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

# Creating a DANK Image
bestText = "TOPPERRRRRRRRR KEK!"
size = (500, 500)
color = (0, 100, 0)
img = Image.new(mode="RGB", size=size, color=color)
imgDrawer = ImageDraw.Draw(img)
font = ImageFont.truetype("roboto.ttf", 30)
imgDrawer.text((20, 250), bestText, font=font)
img.show()
