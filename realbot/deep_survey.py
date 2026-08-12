#!/usr/bin/env python3
"""Dense visual survey of one chunk: N evenly-spread RGB frames (via the index, no full
read) at legible resolution, to catalog localizable objects.
    python realbot/deep_survey.py <chunk.mcap> <N> <out_name>"""
import os, sys
from pathlib import Path
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from PIL import Image, ImageDraw, ImageFont

MCAP = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 15
NAME = sys.argv[3] if len(sys.argv) > 3 else "deep"
OUT = Path(os.environ.get("REALBOT_OUT", str(Path(__file__).resolve().parent / "_out")))
OUT.mkdir(parents=True, exist_ok=True)
RGB = "/zed/zed_node/rgb/color/rect/image"
TW, TH, COLS = 424, 265, 5
fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)

reader = make_reader(open(MCAP, "rb"), decoder_factories=[DecoderFactory()])
st = reader.get_summary().statistics
t0, t1 = st.message_start_time, st.message_end_time
fracs = np.linspace(0.03, 0.97, N)
tiles = []
for fr in fracs:
    tt = int(t0 + (t1 - t0) * fr)
    for s, ch, m, ros in reader.iter_decoded_messages(topics=[RGB], start_time=tt):
        rgb = np.frombuffer(ros.data, np.uint8).reshape(ros.height, ros.width, 4)[:, :, [2, 1, 0]]
        im = Image.fromarray(rgb).resize((TW, TH))
        ImageDraw.Draw(im).text((5, 4), f"{fr:.2f}", fill=(255, 255, 40), font=fnt)
        tiles.append(im); break

rows = (len(tiles) + COLS - 1) // COLS
sheet = Image.new("RGB", (TW * COLS, TH * rows), (20, 20, 28))
for i, im in enumerate(tiles):
    sheet.paste(im, ((i % COLS) * TW, (i // COLS) * TH))
sheet.save(OUT / f"{NAME}.png")
print(f"{NAME}: {len(tiles)} frames -> {sheet.size}")
