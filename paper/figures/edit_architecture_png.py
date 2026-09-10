#!/usr/bin/env python3
"""Pixel-level edits on the v1 ImageGen architecture draft.

Three requested changes (everything else is copied verbatim from the draft):
  1. add the missing stage badge "3" (the draft only had 1, 2, 4);
  2. centre the purple Temporal Transformer box under the red dashed frame;
  3. replace the 3-bend detour from the Temporal Transformer to the fusion
     diamond by a single-bend L route; the "Cross-Attention Fusion" caption
     that used to squeeze it is translated straight down;
  4. lift the fusion diamond by 7 px so that its axis coincides with the
     y=374 spine shared by the three downstream arrows, the ADMM box and the
     "Traffic Allocation" label (the draft had the diamond 7 px too low, so its
     output left the upper-right edge instead of the right apex);
  5. delete the redundant second spatial arrow into the left apex (spatial now
     enters from the top, temporal from the bottom, output leaves to the right);
  6. redraw the upper half of the grey dashed separator, where the draft had
     merged several dashes into two solid 59/64 px strokes and dropped one.

Source  : architecture-draft.png   (untouched)
Output  : architecture-v1-edited.png
"""
from PIL import Image, ImageDraw, ImageFont
import numpy as np

SRC = 'architecture-draft.png'
DST = 'architecture-v1-edited.png'

im = Image.open(SRC).convert('RGB')
arr = np.array(im).astype(int)
R, G, B = arr[..., 0], arr[..., 1], arr[..., 2]
WHITE = (255, 255, 255)


def bbox(mask, x0, x1, y0, y1, label):
    ys, xs = np.nonzero(mask[y0:y1, x0:x1])
    b = (x0 + xs.min(), x0 + xs.max(), y0 + ys.min(), y0 + ys.max())
    print(f'{label}: x {b[0]}-{b[1]}  y {b[2]}-{b[3]}')
    return b


# ---------------------------------------------------------------- measurements
purple = (R > 100) & (R < 245) & (B > 110) & (R - G > 18) & (B - G > 12)
px0, px1, py0, py1 = bbox(purple, 500, 885, 505, 645, 'purple box')

red = (R > 140) & (G < 70) & (B < 80)
rc = red.sum(0)
frame_l = int(np.argmax(rc[200:400]) + 200)
frame_r = int(np.argmax(rc[800:950]) + 800)
shift = int(round((frame_l + frame_r) / 2 - (px0 + px1) / 2))
print(f'red frame verticals: {frame_l}, {frame_r} -> box shift {shift}')

black = (R < 110) & (G < 110) & (B < 110)
lx0, lx1, ly0, ly1 = bbox(black, 880, 1029, 505, 660, 'fusion caption')

# ------------------------------------------------------------------- constants
PAD = 3
CAP_DY = 58                     # caption slides straight down
ARROW_Y = 562                   # y of the new horizontal segment
VX = 957                        # x of the new vertical segment (diamond apex)
STUB_Y0, STUB_Y1 = 566, 584     # rows the old stub covered on the box edge
CLEAN_ROW = 556                 # a stub-free row of the same box edge
SPINE = 374                     # y of the downstream arrow spine
DIA_AXIS = 381                  # y of the diamond axis in the draft
DIA_RECT = (895, 312, 1004, 442)   # diamond + its three arrowheads
DIA_DY = SPINE - DIA_AXIS       # lift needed to put the diamond on the spine
DIA_BOT = 429                   # bottom arrowhead base after the lift
APEX_X = 992                    # x of the diamond right apex
MIRROR = 1915                   # x_left = MIRROR - x_right (axis at 957.5)
GX0, GX1 = 1026, 1039           # x band of the grey dashed separator
GREY_TOP, GREY_SPLIT = 142, 402  # keep the correct lower half from GREY_SPLIT
DASH_PERIOD, DASH_LEN = 24.56, 19

box_rect = (px0 - PAD, py0 - PAD, px1 + PAD + 1, py1 + PAD + 1)
head_rect = (528, 560, 566, 590)             # history arrowhead + some shaft
cap_rect = (lx0 - 2, ly0 - 4, lx1 + 3, ly1 + 4)

# ------------------------------------------------- blocks that have to travel
bb = np.array(im.crop(box_rect)).astype(int)
for y in range(STUB_Y0, STUB_Y1):            # heal the old stub on the right
    for x in range(px1 - 3, box_rect[2]):    # edge by cloning a clean row
        bb[y - box_rect[1], x - box_rect[0]] = arr[CLEAN_ROW, x]
box_blk = Image.fromarray(bb.astype(np.uint8))

head_blk = im.crop(head_rect)

cb = np.array(im.crop(cap_rect)).astype(int)
band = 917 - cap_rect[0]                     # left band: dash + old vertical
sub = cb[:, :band]
spread = sub.max(2) - sub.min(2)
rows = np.arange(cap_rect[1], cap_rect[3])[:, None]
sub[(spread >= 5) & (rows < 585)] = 255      # no glyphs above y=585 there
sub[(sub[..., 0] - sub[..., 1] >= 12) & (rows >= 585)] = 255   # dash only
cb[:, :band] = sub
cap_blk = Image.fromarray(cb.astype(np.uint8))

# dashed frame pixels (core + antialiasing) that the caption move would eat
frm = (R - G > 35) & (R - B > 25)
frame_patch = [(x, y, tuple(arr[y, x]))
               for y in range(cap_rect[1] - 1, cap_rect[3] + CAP_DY + 2)
               for x in range(cap_rect[0] - 1, cap_rect[2] + 2)
               if frm[y, x]]
print('dashed-frame pixels restored:', len(frame_patch))

# ------------------------------------------------------------------- operation
out = im.copy()
draw = ImageDraw.Draw(out)

# (a) clear the old purple box, the old history arrowhead and the old caption
draw.rectangle(box_rect, fill=WHITE)
draw.rectangle((head_rect[0] + shift, head_rect[1],
                head_rect[2], head_rect[3]), fill=WHITE)
draw.rectangle(cap_rect, fill=WHITE)

# (b) clear the old 3-bend route: plain wipe where nothing else lives, and a
#     blue-only wipe where it crossed the dashed frame
draw.rectangle((896, 442, 966, 600), fill=WHITE)
o = np.array(out).astype(int)
zone = np.zeros(o.shape[:2], bool)
zone[558:592, 872:897] = True
o[((o[..., 2] - o[..., 0]) >= 3) & (o.sum(2) < 748) & zone] = 255
out = Image.fromarray(o.astype(np.uint8))
draw = ImageDraw.Draw(out)

# (c) put the travelling pieces back
out.paste(box_blk, (box_rect[0] + shift, box_rect[1]))
out.paste(head_blk, (head_rect[0] + shift, head_rect[1]))
out.paste(cap_blk, (cap_rect[0], cap_rect[1] + CAP_DY))
for x, y, c in frame_patch:
    draw.point((x, y), fill=c)

# (c2) lift the diamond onto the spine.  Its outgoing shaft is tangled with the
#      upper-right edge, so that edge is first rebuilt by mirroring the clean
#      lower-right edge about the diamond axis -- which also wipes the shaft.
oarr = np.array(out)
for y in range(SPINE - 1, SPINE + 2):
    oarr[y, APEX_X - 16:APEX_X + 20] = oarr[2 * DIA_AXIS - y, APEX_X - 16:APEX_X + 20]
dia = Image.fromarray(oarr[DIA_RECT[1]:DIA_RECT[3], DIA_RECT[0]:DIA_RECT[2]])
draw.rectangle(DIA_RECT, fill=WHITE)
out.paste(dia, (DIA_RECT[0], DIA_RECT[1] + DIA_DY))
draw = ImageDraw.Draw(out)

# (c3) re-grow the outgoing shaft, now leaving the right apex itself
o = np.array(out)
sprof = arr[SPINE - 3:SPINE + 4, 1020].astype(np.uint8)
for x in range(APEX_X - 2, 1012):
    for k, c in enumerate(sprof):
        if int(c.sum()) < 735:
            o[SPINE - 3 + k, x] = c
out = Image.fromarray(o)
draw = ImageDraw.Draw(out)

# (c4) drop the redundant second spatial arrow.  Its head is fused with the left
#      apex, so that apex is rebuilt from the mirror image of the clean right
#      half; the mirrored shaft is then wiped together with the original one.
o = np.array(out)
for x in range(915, 951):
    o[355:396, x] = o[355:396, MIRROR - x]
o[355:396, 896:923] = 255
seg = arr[355:396, 888:897]
navy = ((seg[..., 2] - seg[..., 0] > 15) & (seg[..., 2] > 85))[..., None]
o[355:396, 888:897] = np.where(navy, 255, seg).astype(np.uint8)

# (c5) redraw the upper half of the grey dashed separator with the rhythm of
#      its (correct) lower half, then put the crossing shaft back on top
dash = o[500:500 + DASH_LEN, GX0:GX1].copy()
o[GREY_TOP - 4:GREY_SPLIT, GX0:GX1] = 255
for k in range(1, 12):
    s = int(round(GREY_SPLIT - DASH_PERIOD * k))
    if s + DASH_LEN <= GREY_TOP:
        continue
    if s < GREY_TOP:
        o[GREY_TOP:s + DASH_LEN, GX0:GX1] = dash[GREY_TOP - s:]
    else:
        o[s:s + DASH_LEN, GX0:GX1] = dash
for x in range(GX0, GX1):
    for k, c in enumerate(sprof):
        if int(c.sum()) < 735:
            o[SPINE - 3 + k, x] = c
out = Image.fromarray(o)
draw = ImageDraw.Draw(out)

# (d) redraw the connector as a single L: box -> right -> up into the diamond
o = np.array(out)
hprof = arr[571:577, 350].astype(np.uint8)       # horizontal line cross-cut
vprof = arr[310, 956:960].astype(np.uint8)       # vertical line cross-cut
for x in range(px1 + shift - 1, VX + 3):
    for k, c in enumerate(hprof):
        if int(c.sum()) < 735:
            o[ARROW_Y - 2 + k, x] = c
for y in range(DIA_BOT, ARROW_Y + 3):
    for k, c in enumerate(vprof):
        if int(c.sum()) < 735:
            o[y, VX - 1 + k] = c
out = Image.fromarray(o)
draw = ImageDraw.Draw(out)

# (e) the missing stage badge "3", styled like 1/2/4 in the temporal colour
font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 33)
draw.text((986, 299 + DIA_DY), '3', font=font, fill=(126, 46, 84))

# (f) replace the ASCII "(x)" in the header "O = D (x) M" with a proper
#     element-wise product symbol (odot, circle with a centred dot) so the
#     figure matches the formulation notation O^t = D^t \odot M.  The glyphs
#     span x 150-182, y 167-194 (caps 167-188); the D..M gap is centred at
#     x=166, and rows 195-205 are clean white above the grid border at y=206.
draw.rectangle((147, 164, 185, 197), fill=WHITE)     # wipe the old "(x)"
OX, OY, OR = 166, 178, 11                            # centre + radius of odot
draw.ellipse((OX - OR, OY - OR, OX + OR, OY + OR), outline=(0, 0, 0), width=3)
ODOT = 3
draw.ellipse((OX - ODOT, OY - ODOT, OX + ODOT, OY + ODOT), fill=(0, 0, 0))

# (g) MLU 分支没有 ADMM：只重绘右侧后处理区域，保留其余布局。
# 放大绘制后再缩小，保证标签和连线抗锯齿；该步骤可由底图重复生成。
PATCH_X, PATCH_Y = 1117, 325
SCALE = 4
patch = Image.new('RGB', ((out.width - PATCH_X) * SCALE, 128 * SCALE), WHITE)
pd = ImageDraw.Draw(patch)
navy = (27, 64, 81)
blue = (68, 119, 145)
pd.rectangle((20*SCALE, 14*SCALE, 141*SCALE, 86*SCALE),
             fill=(243, 246, 247), outline=navy, width=3*SCALE)
for x0, x1 in ((0, 20), (141, 163)):
    pd.line((x0*SCALE, 49*SCALE, (x1-5)*SCALE, 49*SCALE),
            fill=blue, width=3*SCALE)
    pd.polygon(((x1*SCALE, 49*SCALE), ((x1-9)*SCALE, 44*SCALE),
                ((x1-9)*SCALE, 54*SCALE)), fill=blue)
label_font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 22*SCALE)
for text, x, y in (('Neighbor', 80, 37), ('Refinement', 80, 62),
                   ('Traffic', 205, 37), ('Allocation', 205, 62)):
    pd.text((x*SCALE, y*SCALE), text, font=label_font,
            fill=(0, 0, 0), anchor='mm')
optional_font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 21*SCALE)
pd.text((80*SCALE, 108*SCALE), '(optional)', font=optional_font,
        fill=(70, 70, 70), anchor='mm')
out.paste(patch.resize((out.width - PATCH_X, 128), Image.Resampling.LANCZOS),
          (PATCH_X, PATCH_Y))

out.save(DST)
print('written', DST, out.size)
