from PIL import Image, ImageEnhance, ImageFilter

def blend(gen_img, noback_img):
    width, height = gen_img.size
    mask = erode_mask(noback_img, width, height)
    gen_img.paste(noback_img, mask=mask)
    return gen_img


def erode_mask(noback, width, height):
  mask = Image.new("RGBA", (width,height))
  enhancer = ImageEnhance.Brightness(noback)
  white_object = enhancer.enhance(100)
  mask.paste(white_object, mask=white_object)

  mask = mask.filter(ImageFilter.MinFilter(11))
  return mask