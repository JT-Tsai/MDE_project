from PIL import Image
from PIL.ExifTags import TAGS
img = Image.open("../images/2153117003.jpg")

print(img)
exif = img._getexif()
print(exif)

if exif is not None:
    for (tag, value) in exif.items():
        key = TAGS.get(tag, tag)
        print(key + ' = ' + str(value))
