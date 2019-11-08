"""ONLY FOR DEBUGGING DIFFERENT IMAGE PROCESSING CODE

Not meant to be used in the actual dvrk experiments.
"""
import cv2
import sys
import numpy as np


def process(fname):
    print('processing: {}'.format(fname))
    img = cv2.imread(fname)
    print('  ', img.shape)
    img_01 = cv2.resize(img, (100,100))
    new_fname = fname.replace('.jpg','_downsampled.jpg')
    cv2.imwrite(new_fname, img_01) # looks very pixelated

    # Try a blur
    kernel = np.ones((5,5), np.float32) / 25.0
    img_02 = cv2.filter2D(img, -1, kernel)
    img_02 = cv2.resize(img_02, (100,100))
    new_fname = fname.replace('.jpg','_downsampled_blur.jpg')
    cv2.imwrite(new_fname, img_02)

    # Another blur
    img_03 = cv2.bilateralFilter(src=img, d=9, sigmaColor=75, sigmaSpace=75)
    new_fname = fname.replace('.jpg','_bilateral.jpg')
    cv2.imwrite(new_fname, img_03)
    img_03 = cv2.resize(img_03, (100,100))
    new_fname = fname.replace('.jpg','_downsampled_bilateral.jpg')
    cv2.imwrite(new_fname, img_03)


if __name__ == "__main__":
    process('test_case_01.jpg')
    process('test_case_02.jpg')
    process('test_case_03.jpg')
