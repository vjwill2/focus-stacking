# focus_stacking.py

# Implementation of focus stacking algorithm inspired by the approach described in the paper:
# "Focus Stacking: Bringing Details into Focus," 
# accessible at https://bznick98.github.io/project/focus-stacking

# Victor Will - 12/6/2023

import os
import math
import numpy as np
import skimage
import cv2
from utils import normalize_array, rgb_to_gray
from PIL import Image, ImageEnhance

def build_laplacian_pyramid(gaussian_pyramid):
    """
    Build a Laplacian pyramid from a Gaussian pyramid.

    Parameters:
    gaussian_pyramid (list): A list of numpy.ndarray, each being a level of the Gaussian pyramid.

    Returns:
    list: A list of numpy.ndarray, each being a level of the Laplacian pyramid.
    """
    laplacian_pyramid = []
    levels = len(gaussian_pyramid)

    for i in range(levels - 1):
        # Get the size of the current Gaussian level
        size = (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])

        # Upsample and blur the next level to match the current level size
        gaussian_expanded = cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=size)

        # Subtract the expanded image from the current level to get the Laplacian level
        laplacian = cv2.subtract(gaussian_pyramid[i], gaussian_expanded)
        laplacian_pyramid.append(laplacian)

    # Append the last level of the Gaussian pyramid (the smallest image) to the Laplacian pyramid
    laplacian_pyramid.append(gaussian_pyramid[-1])

    return laplacian_pyramid

def build_gaussian_pyramid(image, levels=3):
    """
    Build a Gaussian pyramid for a given image.

    Parameters:
    image (numpy.ndarray): The input image.
    levels (int): The number of levels in the pyramid. Default is 3.

    Returns:
    list: A list of numpy.ndarray, each being a level of the Gaussian pyramid.
    """
    pyramid = [image]
    for _ in range(1, levels):
        # Apply Gaussian blur
        blurred = cv2.pyrDown(pyramid[-1])
        pyramid.append(blurred)

    return pyramid


def combine_pyramids(pyramids):
    """
    Combine multiple Laplacian pyramids into a single pyramid.

    Parameters:
    pyramids (list): A list of Laplacian pyramids.

    Returns:
    list: A Laplacian pyramid combined from the input pyramids.
    """
    num_pyramids = len(pyramids)
    num_levels = len(pyramids[0])
    combined_pyramid = []

    for level in range(num_levels):
        # Extract the same level from all pyramids
        level_images = [p[level] for p in pyramids]

        # If it's the smallest level, simply average it
        if level == num_levels - 1:
            combined_level = np.mean(level_images, axis=0)
        else:
            # Initialize the mask to zeros
            mask = np.zeros_like(level_images[0], dtype=np.float32)

            # Compute focus measure (e.g., variance of Laplacian) for each image at this level
            for img in level_images:
                laplacian_var = cv2.Laplacian(img, cv2.CV_32F).var()
                mask = np.maximum(mask, laplacian_var)

            # Normalize the mask
            mask /= mask.max()

            # Combine images based on the mask
            combined_level = sum(img * mask for img in level_images)

        combined_pyramid.append(combined_level.astype(level_images[0].dtype))

    return combined_pyramid


def reconstruct_image_from_pyramid(laplacian_pyramid):
    """
    Reconstruct an image from a Laplacian pyramid.

    Parameters:
    laplacian_pyramid (list): A Laplacian pyramid as a list of numpy.ndarray.

    Returns:
    numpy.ndarray: The reconstructed image.
    """
    # Start with the smallest image at the top of the pyramid
    reconstructed_image = laplacian_pyramid[-1]

    # Progressively expand and add details from each level
    for level in range(len(laplacian_pyramid) - 2, -1, -1):
        size = (laplacian_pyramid[level].shape[1], laplacian_pyramid[level].shape[0])
        reconstructed_image = cv2.pyrUp(reconstructed_image, dstsize=size)
        reconstructed_image = cv2.add(reconstructed_image, laplacian_pyramid[level])

    return reconstructed_image


def adjust_contrast_and_color(image, contrast_factor=1.5, color_factor=1.2):
    """
    Adjust the contrast and color of an image.

    Parameters:
    image (numpy.ndarray): The input image.
    contrast_factor (float): Factor to adjust contrast. >1 increases contrast.
    color_factor (float): Factor to adjust color/saturation. >1 increases saturation.

    Returns:
    numpy.ndarray: The adjusted image.
    """

    # Normalize the image to 0-255 and convert to uint8
    norm_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)

    # Convert to PIL Image for easier adjustments
    image_pil = Image.fromarray(cv2.cvtColor(norm_image, cv2.COLOR_BGR2RGB))

    # Adjust contrast
    enhancer = ImageEnhance.Contrast(image_pil)
    image_pil = enhancer.enhance(contrast_factor)

    # Adjust color (saturation)
    enhancer = ImageEnhance.Color(image_pil)
    image_pil = enhancer.enhance(color_factor)

    # Convert back to OpenCV format
    image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    return image_cv


def align_images(images: np.array, num_iter=1000, term_eps=1e-5, warp_mode=cv2.MOTION_HOMOGRAPHY) -> np.array:
    '''
    Aligns images using the ECC algorithm

    images - numpy array representing list of images to align
    num_iter - maximum number of alignment iterations
    term_eps - epsilon value for min change between alignment iterations (if less, terminates)
    warp_mode - type of transformation to calculate alignment with

    returns - numpy array of aligned images
    '''
    assert len(images) > 1  # At least 2 images to align
    assert len(images.shape) == 4  # Expect images in color
    assert images.shape[3] == 3
    assert warp_mode in [cv2.MOTION_HOMOGRAPHY,  # Possible warp modes
                         cv2.MOTION_EUCLIDEAN,
                         cv2.MOTION_AFFINE,
                         cv2.MOTION_HOMOGRAPHY]

    ref_img = images[0]
    ref_gray = rgb_to_gray(ref_img)
    height, width, _ = ref_img.shape

    ref_corners = np.array([
        [0,       0,        1],
        [width-1, 0,        1],
        [0,       height-1, 1],
        [width-1, height-1, 1],
    ])

    aligned = [ref_img]
    new_corners = [ref_corners[:, :-1]]

    for i, img in enumerate(list(images[1:])):
        print(f"Align Image {i + 2}")
        img_gray = rgb_to_gray(img)

        # Define 2x3 or 3x3 matrices and initialize the matrix to identity
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS |
                    cv2.TERM_CRITERIA_COUNT, num_iter, term_eps)

        # Run the ECC algorithm. The results are stored in warp_matrix.
        _, warp_matrix = cv2.findTransformECC(
            ref_gray, img_gray, warp_matrix, warp_mode, criteria)

        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            # Use warpPerspective for Homography
            img_aligned = cv2.warpPerspective(
                img, warp_matrix, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else:
            # Use warpAffine for Translation, Euclidean and Affine
            img_aligned = cv2.warpAffine(
                img, warp_matrix, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

        if warp_mode != cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.vstack((warp_matrix, np.array([0, 0, 1])))

        # Calculate corners of warped image
        inv_warp = np.linalg.inv(warp_matrix)
        corners = ref_corners @ inv_warp.T
        corners = (corners.T / corners[:, -1])[:-1].T

        new_corners.append(corners)
        aligned.append(img_aligned)

    # Calculate and apply crop
    new_corners = np.array(new_corners)
    tl_x = new_corners[:, 0, 0]
    tl_y = new_corners[:, 0, 1]
    tr_x = new_corners[:, 1, 0]
    tr_y = new_corners[:, 1, 1]
    bl_x = new_corners[:, 2, 0]
    bl_y = new_corners[:, 2, 1]
    br_x = new_corners[:, 3, 0]
    br_y = new_corners[:, 3, 1]

    tl_x = math.ceil(np.max([tl_x, bl_x]))
    tl_y = math.ceil(np.max([tl_y, tr_y]))
    br_x = math.floor(np.min([br_x, tr_x]))
    br_y = math.floor(np.min([br_y, bl_y]))

    # Crop image
    aligned = [img[tl_y:br_y, tl_x:br_x] for img in aligned]

    return np.array(aligned)


def focus_stacking(images):
    """
    Performs focus stacking on a list of images to create a single image with extended depth of field.

    Steps:
    1. Align images using ECC algorithm.
    2. Build Gaussian and Laplacian pyramids for each aligned image.
    3. Combine these pyramids to focus on the most detailed parts of each image.
    4. Reconstruct the final image from the combined pyramid.
    5. Apply post-processing for contrast and color enhancement.

    Parameters:
    images (list of numpy.ndarray): List of images for focus stacking.

    Returns:
    numpy.ndarray: Focus-stacked image with improved depth of field and sharpness.
    """

    aligned_images = align_images(images)
    gaussian_pyramids = [build_gaussian_pyramid(img, levels=3) for img in aligned_images]
    laplacian_pyramids = [build_laplacian_pyramid(gpyr) for gpyr in gaussian_pyramids]
    combined_pyramid = combine_pyramids(laplacian_pyramids)
    focused_image = reconstruct_image_from_pyramid(combined_pyramid)
    adjusted_image = adjust_contrast_and_color(focused_image)

    return adjusted_image