import os
import numpy as np
import matplotlib.pyplot as plt

from naive_approach import naive_blend
from depth_mapping import depth_mapping, DepthMapArgs
from utils import load_image, plt_img, save_fig, save_img


def main():
    #################### Ant Results ####################
    # Load images
    name = 'ant'
    img_dir = './source_images/ant'
    img_names = [
        'b_bigbug0001_croppped.png', 'b_bigbug0002_croppped.png', 'b_bigbug0003_croppped.png', 'b_bigbug0004_croppped.png',
        'b_bigbug0005_croppped.png', 'b_bigbug0006_croppped.png', 'b_bigbug0007_croppped.png', 'b_bigbug0008_croppped.png',
        'b_bigbug0009_croppped.png', 'b_bigbug0010_croppped.png', 'b_bigbug0011_croppped.png', 'b_bigbug0012_croppped.png'
    ]
    images = np.array([load_image(os.path.join(img_dir, name))
                      for name in img_names])
    # Set focal depths
    focal_depths = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    ##### Naive Focus Stacking #####
    gen_results_naive(name, images, alpha=0.5)

    ##### Laplacian Pyramid Focus Stacking Results #####

    ##### Depth Mapping Results #####
    # Configure arguments
    args = DepthMapArgs()
    args.bg_thresh = 0.5
    args.map_max_pool_size = 1
    gen_results_depth(name, images, focal_depths, args)

    #################### Bug Results ####################
    # Load images
    name = 'bug'
    img_dir = './source_images/bug'
    img_names = [
        'hf001.jpg', 'hf005.jpg', 'hf010.jpg', 'hf015.jpg',
        'hf020.jpg', 'hf025.jpg', 'hf030.jpg', 'hf035.jpg',
    ]
    images = np.array([load_image(os.path.join(img_dir, name))
                      for name in img_names])
    # Set focal depths
    focal_depths = np.array([1, 2, 3, 4, 5, 6, 7, 8])

    ##### Naive Focus Stacking #####
    gen_results_naive(name, images, alpha=0.5)

    ##### Laplacian Pyramid Focus Stacking Results #####

    ##### Depth Mapping Results #####
    # Configure arguments
    args = DepthMapArgs()
    args.bg_thresh = 0.15
    args.map_max_pool_size = 3
    gen_results_depth(name, images, focal_depths, args)


def gen_results_naive(name, images, alpha):
    '''
    Generates naive blending results for given images
    '''
    print(f"\nNaive Blending: {name}")
    stack = naive_blend(images, alpha)

    save_img(f"./results_naive/{name}.png", stack)


def gen_results_laplacian(name, images):
    '''
    Generates laplacian blending results for given images
    '''
    print(f"\nLaplacian Blenlding: {name}")
    stack = images # call stacking function

    save_img(f"./results_laplacian/{name}.png", stack)


def gen_results_depth(name, images, focal_depths, args):
    '''
    Generates depth mapping results for given images
    '''
    print(f"\nDepth Mapping: {name}")
    depth, stack, intermediates = depth_mapping(images, focal_depths, args)
    aligned, sharpness, bg_mask = intermediates

    # Save results
    save_img(f'./results_stack/{name}/stack.png', stack)
    save_img(f'./results_stack/{name}/depth.png', depth)

    fig, ax = plt.subplots(1, len(aligned), figsize=(len(aligned) * 10, 10))
    for i in range(len(aligned)):
        plt_img(ax[i], aligned[i])
    save_fig(f'./results_stack/{name}/aligned.png', fig)
    plt.close(fig)

    fig, ax = plt.subplots(
        1, len(sharpness), figsize=(len(sharpness) * 10, 10))
    for i in range(len(sharpness)):
        plt_img(ax[i], sharpness[i])
    save_fig(f'./results_stack/{name}/sharpness.png', fig)
    plt.close(fig)

    fig, ax = plt.subplots(1)
    plt_img(ax, bg_mask)
    save_fig(f'./results_stack/{name}/bg_mask.png', fig)
    plt.close(fig)


if __name__ == '__main__':
    main()
