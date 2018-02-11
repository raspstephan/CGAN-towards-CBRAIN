"""
Utility functions.
"""
import numpy as np
from PIL import Image


def load_all_imgs(files, img_size):
    """Loads all facades images
    """
    a_list = []
    b_list = []
    for fn in files:
        img_arr = np.array(Image.open(fn))
        a_list.append(img_arr[:, :img_size])
        b_list.append(img_arr[:, img_size:])
    return np.array(a_list), np.array(b_list)