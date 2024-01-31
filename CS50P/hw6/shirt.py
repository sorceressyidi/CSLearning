import os
from PIL import Image,ImageOps
import sys
shirt = Image.open("shirt.png")
types = (".jpg", ".jpeg", ".png")
if (
    len(sys.argv) != 3
    or not sys.argv[1].endswith(types)
    or not sys.argv[2].endswith(types)
    or sys.argv[1].split(".")[1] != sys.argv[2].split(".")[1]
):
    sys.exit("Invalid arguments.")
input = sys.argv[1]
output = sys.argv[2]
try :
    with Image.open(input) as image:
        image = ImageOps.fit(image, shirt.size)
        image.paste(shirt, shirt)
        image.save(output)
except FileNotFoundError:
    sys.exit("File does not exist.")