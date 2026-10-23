"""Builds logo_full_{light,dark}.svg (AGE logo + "NOMAD Oasis") for the new GUI.

nginx serves them in place of NOMAD's bundled nomad_logo_full_*.svg. The text is
converted to paths because an SVG shown via <img> cannot load web fonts.

    pip install pillow fonttools
    python make_logo.py path/to/age.png path/to/Roboto-Bold.ttf
"""

import base64
import io
import sys

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from PIL import Image

LOGO, FONT = sys.argv[1], sys.argv[2]
TEXT, FONT_SIZE, GAP = 'NOMAD Oasis', 380, 110
COLORS = {'dark': (139, 140, 236), 'light': (25, 46, 134)}  # NOMAD palette

font = TTFont(FONT)
glyphs, cmap, upm = font.getGlyphSet(), font.getBestCmap(), font['head'].unitsPerEm
scale = FONT_SIZE / upm
pen, x = SVGPathPen(glyphs), 0
for ch in TEXT:
    name = cmap[ord(ch)]
    glyphs[name].draw(TransformPen(pen, (scale, 0, 0, -scale, x * scale, 0)))
    x += glyphs[name].width
text_width, cap = x * scale, font['OS/2'].sCapHeight * scale

logo = Image.open(LOGO).convert('RGBA')
logo = logo.crop(logo.getbbox())
W, H = logo.size

for theme, rgb in COLORS.items():
    tinted = Image.new('RGBA', logo.size, (*rgb, 255))
    tinted.putalpha(logo.getchannel('A'))
    buf = io.BytesIO()
    tinted.save(buf, 'PNG', optimize=True)
    total = W + GAP + text_width
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'viewBox="0 0 {total:.0f} {H}" width="{total:.0f}" height="{H}">\n'
        f'<image x="0" y="0" width="{W}" height="{H}" xlink:href="data:image/png;'
        f'base64,{base64.b64encode(buf.getvalue()).decode()}"/>\n'
        f'<path transform="translate({W + GAP} {H / 2 + cap / 2:.0f})" '
        f'fill="#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}" d="{pen.getCommands()}"/>\n'
        '</svg>\n'
    )
    with open(f'logo_full_{theme}.svg', 'w') as f:
        f.write(svg)
