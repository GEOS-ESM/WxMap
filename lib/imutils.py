from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageChops

def im_draw_places(im, places, font_name, font_size, font_color=(255,255,255), shadow_color=(0,0,0), position='c'):

    d1 = HersheyDraw(im, font_name, font_size, shadow_color)
    d2 = HersheyDraw(im, font_name, font_size, font_color)

    for place in places:
        p = place.split()
        xo = float(p[0])
        yo = float(p[1])
        name = ' '.join(p[2:])
        w, h = d1.text_size(name)
        x, y = str_position(xo, yo, w, h, position)
     #  x = xo-w/2
     #  y = yo-h/2

        for xdelta in [-2,-1,1,2]:
            for ydelta in [-2,-1,1,2]:
                d1.draw_text(x+xdelta, y+ydelta, name)

        d2.draw_text(x, y, name)

def str_position(x, y, w, h, pos):

    if pos == 'c':
        return (x-w/2, y-h/2)
    elif pos == 'tl':
        return (x, y)
    elif pos == 'bl':
        return (x, y-h)

def im_paste_file(bg, fname, xpos, ypos, xsize=None, ysize=None):

    img     = Image.open(fname)
    
    if xsize:
        ratio   = float(img.size[1]) / float(img.size[0])
        ysize   = int(xsize * ratio)
    
    elif ysize:
        ratio   = float(img.size[0]) / float(img.size[1])
        xsize   = int(ysize * ratio)
    
    if xsize and ysize:
        img = img.resize((xsize, ysize), Image.Resampling.LANCZOS)

    xsize = img.size[0]
    ysize = img.size[1]
    
    bg.paste(img, (xpos, ypos), img)
    
    img.close()

    return (xsize, ysize)

def image_trim(im, margin=0):

    bg = Image.new(im.mode, im.size, im.getpixel((0,0))).convert('RGB')
    diff = ImageChops.difference(im.convert('RGB'), bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = list(diff.getbbox())
    if bbox:
        bbox[0] -= margin
        bbox[1] -= margin
        bbox[2] += margin
        bbox[3] += margin
        return im.crop(bbox)
    return im

class HersheyDrawOld(object):
    """
    Defines methods for drawing formatted text on an image.

    The methods in this class use the PIL library to draw text
    on an image. The text may contain format control characters
    as defined by the GrADS Hershey font for super- and sub-scripting.

    Requirements
    ------------
    from PIL import ImageDraw, ImageFont
    """

    def __init__(self, im, font, size, color, **kwargs):
        """
        Initializer

        Parameters
        ----------
        im : Image
            PIL image object.
        font : string
            Name of font file (Truetype etc.).
        size : integer
            Font size (pixels). Super and sub-scripting font size
            is set to size/2.
        color : tuple
            RGB(A) value for font color (R, G, B, A)

        Returns
        -------
        None
            No return value

        """

        self.color = color
        self.d     = ImageDraw.Draw(im)
        self.fn    = ImageFont.truetype(font, size)
        self.fs    = ImageFont.truetype(font, int(size/2))
        self.hn    = self.d.textsize('1', font=self.fn)[1]
        self.hs    = self.d.textsize('1', font=self.fs)[1]

    def text_size(self, text):
        """
        Returns the size of the text string in pixels.

        This method returns the size of the string in pixels after
        resolving the Hershey font control characters.

        Parameters
        ----------
        text : string
            Text string with or without Hershey control characters. Only
            superscripting (`a), subscripting (`b) and normal (`n) formatting
            sequences are recognized.

        Returns
        -------
        size : tuple
            width and height of the text string (w, h) where "h" is the
            maximum height of the string.

        """

        if not text: return (0, 0)

        words = text.split('`')
        if words[0]: words[0] = 'n' + words[0]

        nText, sText = ('', '')

        for s in words:

            if not s: continue
            if len(s) == 1: continue

            if s[0] == 'n':
                nText += s[1:]
            else:
                sText += s[1:]

        wn, hn, ws, hs = (0, 0, 0, 0)
        if nText: wn, hn = self.d.textsize(nText, font=self.fn)
        if sText: ws, hs = self.d.textsize(sText, font=self.fs)

        return (wn + ws, max(hn,hs))

    def draw_text(self, x, y, text):
        """
        Draws text on an image.

        This method uses the PIL library to draw text on the image.
        The text may contain format control characters as defined by
        the GrADS Hershey font for super- and sub-scripting.

        Parameters
        ----------
        x : integer
            Left pixel location of string.

        y : integer
            Top pixel location of string.

        text : string
            Text string with or without Hershey control characters. Only
            superscripting (`a), subscripting (`b) and normal (`n) formatting
            sequences are recognized.

        Returns
        -------
        None
            No return value

        """

        if not text: return

        words = text.split('`')
        if words[0]: words[0] = 'n' + words[0]
 
        for s in words:

            if not s: continue
            if len(s) == 1: continue

            if s[0] == 'n':
                self.d.text( (x, y), s[1:], font=self.fn, fill=self.color)
                w, h = self.d.textsize(s[1:], font=self.fn)
                x += w
            elif s[0] == 'a':
                self.d.text( (x, y), s[1:], font=self.fs,
                                                      fill=self.color)
                w, h = self.d.textsize(s[1:], font=self.fs)
                x += w
            elif s[0] == 'b':
                self.d.text( (x, y+self.hn-self.hs), s[1:], font=self.fs,
                                                      fill=self.color)
                w, h = self.d.textsize(s[1:], font=self.fs)
                x += w

def round_corner(radius, color):
    """Draw a round corner"""
    corner = Image.new('RGBA', (radius, radius), (0, 0, 0, 0))
    draw = ImageDraw.Draw(corner)
    draw.pieslice((0, 0, radius * 2, radius * 2), 180, 270, fill=color)
    return corner

def round_rectangle(size, radius, color):
    """Draw a rounded rectangle"""
    width, height = size
    rectangle = Image.new('RGBA', size, color)

    origCorner = round_corner(radius, color)
    corner = origCorner.rotate(270)
    rectangle.paste(corner, (width - radius, 0))

    return rectangle

class ImageMapper(object):

    def __init__(self, img, bbox):

        width = img.width
        height = img.height

        self.img = img
        self.bbox = tuple(bbox)
        self.x_factor = float(width-1) / (bbox[2] - bbox[0])
        self.y_factor = float(height-1) / (bbox[3] - bbox[1])

    def subset(self, bbox):

        x1 = int(round(float(bbox[0]-self.bbox[0]) * self.x_factor))
        y1 = int(round(float(bbox[1]-self.bbox[1]) * self.y_factor))
        x2 = int(round(float(bbox[2]-self.bbox[0]) * self.x_factor))
        y2 = int(round(float(bbox[3]-self.bbox[1]) * self.y_factor))

        return ImageMapper(self.img.crop((x1, y1, x2, y2)), bbox)

    def resize(self, bbox):
        return ImageMapper(self.img.resize(bbox, Image.Resampling.LANCZOS), self.bbox)

    def to_pixel(self, bbox):

        obox = []

        for i in range(0, len(bbox), 2):
            lon = bbox[i]
            lat = bbox[i+1]
            x = int(round(float(lon - self.bbox[0]) * self.x_factor))
            y = int(round(float(lat - self.bbox[1]) * self.y_factor))
            obox.append(x)
            obox.append(self.img.height-y-1)

        return obox

class HersheyDraw(object):
    """
    Defines methods for drawing formatted text on an image.

    The methods in this class use the PIL library to draw text
    on an image. The text may contain format control characters
    as defined by the GrADS Hershey font for super- and sub-scripting.

    Requirements
    ------------
    from PIL import ImageDraw, ImageFont
    """

    def __init__(self, im, font, size, color, **kwargs):
        """
        Initializer

        Parameters
        ----------
        im : Image
            PIL image object.
        font : string
            Name of font file (Truetype etc.).
        size : integer
            Font size (pixels). Super and sub-scripting font size
            is set to size/2.
        color : tuple
            RGB(A) value for font color (R, G, B, A)

        Returns
        -------
        None
            No return value

        """

        self.color = color
        self.d     = ImageDraw.Draw(im)
        self.fn    = ImageFont.truetype(font, size)
        self.fs    = ImageFont.truetype(font, int(size/2))
        self.hn    = self.get_text_dimensions('1', font=self.fn)[1]
        self.hs    = self.get_text_dimensions('1', font=self.fs)[1]

    def text_size(self, text):
        """
        Returns the size of the text string in pixels.

        This method returns the size of the string in pixels after
        resolving the Hershey font control characters.

        Parameters
        ----------
        text : string
            Text string with or without Hershey control characters. Only
            superscripting (`a), subscripting (`b) and normal (`n) formatting
            sequences are recognized.

        Returns
        -------
        size : tuple
            width and height of the text string (w, h) where "h" is the
            maximum height of the string.

        """

        if not text: return (0, 0)

        words = text.split('`')
        if words[0]: words[0] = 'n' + words[0]

        nText, sText = ('', '')

        for s in words:

            if not s: continue
            if len(s) == 1: continue

            if s[0] == 'n':
                nText += s[1:]
            else:
                sText += s[1:]

        wn, hn, ws, hs = (0, 0, 0, 0)
        if nText: wn, hn = self.get_text_dimensions(nText, font=self.fn)
        if sText: ws, hs = self.get_text_dimensions(sText, font=self.fs)

        return (wn + ws, max(hn,hs))

    def draw_text(self, x, y, text):
        """
        Draws text on an image.

        This method uses the PIL library to draw text on the image.
        The text may contain format control characters as defined by
        the GrADS Hershey font for super- and sub-scripting.

        Parameters
        ----------
        x : integer
            Left pixel location of string.

        y : integer
            Top pixel location of string.

        text : string
            Text string with or without Hershey control characters. Only
            superscripting (`a), subscripting (`b) and normal (`n) formatting
            sequences are recognized.

        Returns
        -------
        None
            No return value

        """

        if not text: return

        words = text.split('`')
        if words[0]: words[0] = 'n' + words[0]
 
        for s in words:

            if not s: continue
            if len(s) == 1: continue

            if s[0] == 'n':
                self.d.text( (x, y), s[1:], font=self.fn, fill=self.color)
                w, h = self.get_text_dimensions(s[1:], font=self.fn)
                x += w
            elif s[0] == 'a':
                self.d.text( (x, y), s[1:], font=self.fs,
                                                      fill=self.color)
                w, h = self.get_text_dimensions(s[1:], font=self.fs)
                x += w
            elif s[0] == 'b':
                self.d.text( (x, y+self.hn-self.hs), s[1:], font=self.fs,
                                                      fill=self.color)
                w, h = self.get_text_dimensions(s[1:], font=self.fs)
                x += w

    def get_text_dimensions(self, text_string, font):

        ascent, descent = font.getmetrics()

        text_width = font.getmask(text_string).getbbox()[2]
        text_height = font.getmask(text_string).getbbox()[3] + descent

        return (text_width, text_height)
