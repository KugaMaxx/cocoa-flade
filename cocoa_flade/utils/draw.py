import random
import platform
import numpy as np
from typing import List
from PIL import Image, ImageDraw, ImageFont


def plot_projected_events(bkg_image, events):
    # convert to pil image
    height, width, *_ = bkg_image.shape
    bkg_image = Image.fromarray(bkg_image).convert('RGBA')

    # classify based on polarity
    _, x, y, polarity = events[:, 0], events[:, 1], events[:, 2], events[:, 3]
    pos_ind = polarity == 1
    neg_ind = polarity == 0

    # project events
    evt_image = Image.new('RGBA', (width, height))
    evt_draw = ImageDraw.Draw(evt_image)
    evt_draw.point(list(zip(x[pos_ind], y[pos_ind])), fill=(0, 0, 255, 128))
    evt_draw.point(list(zip(x[neg_ind], y[neg_ind])), fill=(255, 0, 0, 128))

    # hybrid
    pil_image = Image.alpha_composite(bkg_image, evt_image)

    return np.array(pil_image)


def plot_rescaled_image(bkg_image, factor=2):
    # convert to pil image
    bkg_image = Image.fromarray(bkg_image).convert('RGBA')

    # define rescaled function
    from scipy.ndimage import zoom
    rescale = lambda x: zoom(x, zoom=(factor, factor, 1), order=3)

    return np.array(rescale(bkg_image))


def plot_detection_result(bkg_image, bboxes: List, labels: List = None, scores: List = None,
                          categories=[{
                                  'id': i,
                                  'name': f"cat_{i}",
                                  'color': "#{:02x}{:02x}{:02x}".format(
                                      random.randint(0, 255),
                                      random.randint(0, 255),
                                      random.randint(0, 255),
                                  )
                              } for i in range(100)
                          ]):
    """
    https://github.com/trsvchn/coco-viewer/blob/main/cocoviewer.py
    """
    # pre-processing
    height, width, *_ = bkg_image.shape
    categories = {cat['id']: cat for cat in categories}

    # rescale
    bboxes = [
        [
            x * width,
            y * height,
            x * width + w * width,
            y * height + h * height
        ]
        for (x, y, w, h) in bboxes
    ]

    # color convert function
    hex_to_rgb = lambda hex: tuple(int(hex.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))

    # create PIL image
    pil_image = Image.fromarray(bkg_image).convert("RGBA")
    draw = ImageDraw.Draw(pil_image)
    try:
        try:
            # Linux
            font = ImageFont.truetype("DejaVuSans.ttf", size=int(min(width, height) * 0.08))
        except OSError:
            # Windows
            font = ImageFont.truetype("Arial.ttf", size=int(min(width, height) * 0.08))
    except OSError:
        # Load default, note no resize option
        font = ImageFont.load_default()

    # drawing
    for k, (tl_x, tl_y, rb_x, rb_y) in enumerate(bboxes):
        # obtain element
        label = labels[k] if labels is not None else 0
        proba = scores[k] if scores is not None else None
        if label not in categories.keys(): continue
        name  = categories[label]['name']
        color = categories[label]['color']

        # draw rectangle
        draw.rectangle((tl_x, tl_y, rb_x, rb_y), 
                       width = max(1, int(min(width, height) * 0.015)),
                       outline = hex_to_rgb(color) + (200,))

        # draw text on the image
        text = f"{name}: {proba:.2f}" if proba is not None else f"{name}"
        (text_width, text_height), (_, _) = font.font.getsize(text)
        text_x = tl_x if tl_x + text_width < width else width - text_width
        text_y = tl_y - text_height if tl_y - text_height > 0 else 0
        text_bbox = draw.textbbox((text_x, text_y), text, font=font)

        margin = int(min(width, height) * 0.01)
        text_bbox = (text_bbox[0], text_bbox[1] - margin,
                     text_bbox[2], text_bbox[3] + margin)

        draw.rectangle(text_bbox, fill=hex_to_rgb(color)+(200,))
        draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)

    return np.array(pil_image)
