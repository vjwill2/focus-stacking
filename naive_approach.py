# naive_approach.py

# Implementation of naive focus stacking approach

# Deeya Bodas - 12/6/2023

import numpy as np
import cv2


def naive_blend(imgs, alpha):
    filtered_imgs = []

    # Create a kernel to apply to each image
    kernel = np.ones((5, 5), np.float32)/30

    # Apply a filter to each image
    for img in imgs:
        filtered_imgs.append(cv2.filter2D(img, -1, kernel))

    # Blend the filtered images to simulate a form of "lazy" focus stacking
    # Use the given alpha value for blending

    final_blended_img = filtered_imgs[0]
    for i in range(1, len(filtered_imgs) - 1):
        final_blended_img = cv2.addWeighted(
            final_blended_img, alpha, filtered_imgs[i], 1 - alpha, 0)

    return final_blended_img
