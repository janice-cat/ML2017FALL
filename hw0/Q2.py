import sys
from PIL import Image, ImageDraw
im = Image.open(sys.argv[1])

rgb_im = im.convert('RGB')
r, g, b = 0, 0, 0
w, h = im.size

im2 = Image.new('RGB', (w, h), (255, 255, 255))
draw = ImageDraw.Draw(im2)

for y in range(0,h,1):
	for x in range(0,w,1):
		r, g, b = rgb_im.getpixel((x, y))
		r = r//2
		g = g//2
		b = b//2
		draw.point((x,y),fill = (r,g,b))
im2.save('Q2.png')
