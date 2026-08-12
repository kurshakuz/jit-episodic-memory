#!/usr/bin/env python3
"""Grab a few spread-out RGB frames from every .mcap chunk (via the index, no full read)
to survey the whole run's scenes/objects.

    python realbot/survey_chunks.py <dir-with-mcaps>   (writes survey.png to REALBOT_OUT)
"""
import os, sys
from pathlib import Path
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from PIL import Image, ImageDraw, ImageFont

D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
files = sorted(D.glob("*.mcap"))
RGB = "/zed/zed_node/rgb/color/rect/image"
OUT = Path(os.environ.get("REALBOT_OUT", str(Path(__file__).resolve().parent / "_out")))
OUT.mkdir(parents=True, exist_ok=True)
FR = [0.08, 0.3, 0.5, 0.7, 0.92]
fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)

rows = []
for f in files:
    reader = make_reader(open(f, "rb"), decoder_factories=[DecoderFactory()])
    st = reader.get_summary().statistics
    t0, t1 = st.message_start_time, st.message_end_time
    tiles = []
    for fr in FR:
        tt = int(t0 + (t1 - t0) * fr)
        for schema, ch, message, ros in reader.iter_decoded_messages(topics=[RGB], start_time=tt):
            rgb = np.frombuffer(ros.data, np.uint8).reshape(ros.height, ros.width, 4)[:, :, [2, 1, 0]]
            tiles.append(Image.fromarray(rgb).resize((300, 188))); break
    lab = Image.new("RGB", (130, 188), (15, 16, 26))
    ImageDraw.Draw(lab).text((6, 8), f.stem[-14:], fill=(255, 255, 255), font=fnt)
    row = Image.new("RGB", (130 + 300 * len(tiles), 188), (0, 0, 0)); row.paste(lab, (0, 0))
    for i, im in enumerate(tiles):
        row.paste(im, (130 + i * 300, 0))
    rows.append(row)
    print(f"{f.name}: dur {(t1-t0)/1e9:.0f}s")

W = max(r.width for r in rows); H = sum(r.height + 6 for r in rows)
sheet = Image.new("RGB", (W, H), "white"); y = 0
for r in rows:
    sheet.paste(r, (0, y)); y += r.height + 6
sheet.save(OUT / "survey.png"); print("survey ->", OUT / "survey.png", sheet.size)
