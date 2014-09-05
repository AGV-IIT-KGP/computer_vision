# Install PIL

import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure

plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['image.interpolation'] = 'none'


def draw_h(image, coords, in_place=True):
    height, width, thick = 24, 20, 3
    row, col = coords
    green = [0, 1.0, 0]
    if not in_place:
        image = image.copy()

    image[row: (row + height), col: (col + thick), :] = green
    image[row: (row + height), (col + width - thick): (col + width), :] = green
    strut = row + (height//2)
    image[strut: (strut + thick), col: (col + width), :] = green
    return image


def red_blue_green_images(color_image):
    red_image = np.zeros_like(color_image)
    green_image = np.zeros_like(color_image)
    blue_image = np.zeros_like(color_image)

    red_image[:, :, 0] = color_image[:, :, 0]
    green_image[:, :, 1] = color_image[:, :, 1]
    blue_image[:, :, 2] = color_image[:, :, 2]

    return red_image, green_image, blue_image


def random_stained_glass_squares(color_image, no_squares=10):
    red_image, green_image, blue_image = red_blue_green_images(color_image)
    color_dict = {0: red_image,
                  1: green_image,
                  2: blue_image}

    color_image = color_image.copy()
    rows, cols, channels = color_image.shape

    for i in range(no_squares):
        x1, y1 = np.random.randint(0, rows/2), np.random.randint(0, cols/2)
        x2, y2 = np.random.randint(x1, rows), np.random.randint(y1, cols)

        color_index = np.random.randint(3)

        color_image[x1: x2, y1: y2, :] = \
            color_dict[color_index][x1: x2, y1: y2, :]

    return color_image


def plot_hist(color_image):
    for color, channel in zip('rgb', np.rollaxis(color_image, axis=-1)):
        counts, bin_centers = exposure.histogram(channel)
        # plt.hist(channel.flatten(), color=color, alpha=0.5)
        plt.plot(bin_centers, counts, color=color)


# linearly streching pixel values
# uint8 causes problems
