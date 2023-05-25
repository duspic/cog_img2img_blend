from PIL import Image

def overlay(img):
  width, height = img.size
  res = Image.new("RGB", (width, height),(255,255,255))
  res.paste(img, mask=img)
  
  return res