import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def normalize_array(arr: np.array, uint8_mode=False) -> np.array:
    '''
    Normalizes numpy array to be float32 with range 0-1 or uint8 with range 0-255

    arr - array to normalize
    uint8_mode - select float32 or uint8, defaults to float32

    returns - uint8 or float32 numpy array in same shape as input
    '''
    arr = arr.astype('float32')
    if np.max(arr) - np.min(arr) == 0:
        return arr
    arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    if uint8_mode:
        arr *= 255
        arr = arr.astype('uint8')
    return arr


def load_image(path: os.PathLike) -> np.array:
    '''
    Loads a given image in RGB from given path

    path - path to image

    returns - float32 numpy array representing an RGB image
    '''
    try:
        bgr_img = cv2.imread(path)
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        return normalize_array(rgb_img)
    except:
        raise Exception(f'Failed to load image: {path}')


def rgb_to_gray(im: np.array) -> np.array:
    '''
    Converts a RGB image to grayscale

    im - image to convert

    returns - numpy array representing a grayscale image
    '''
    assert len(im.shape) == 3  # Image must be RGB
    return cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)


def plt_img(ax: plt.axis, img: np.array, title='') -> None:
    '''
    Plots given image onto matplot axis with an optional title

    ax - matplot plot axis (can just pass in matplot.pyplot, or a subplot axis)
    img - grayscale or RGB image to plot
    title - optional plot title
    '''
    cmap = 'gray'
    if (len(img.shape) == 3 and img.shape[2] == 3):
        cmap = None
    ax.imshow(img, cmap=cmap)
    ax.axis('off')
    if title:
        ax.set_title(title)


def save_fig(path: os.PathLike, fig: plt.Figure) -> None:
    '''
    Saves given plot to given path.

    path - path to save plot to
    img - image to save
    '''
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches='tight')
    print(f"Image saved at {path}")


def save_img(path: os.PathLike, img: np.array) -> None:
    '''
    Saves given image to given path. Converts non uint8 images to uint8

    path - path to save image to
    img - image to save
    '''
    if (len(img.shape) == 3):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if (img.dtype != 'uint8'):
        img = normalize_array(img, uint8_mode=True)

    # Ensure output directory exists
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)
    print(f"Image saved at {path}")

