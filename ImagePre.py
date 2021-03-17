import cv2
import numpy as np
# from stardist.models import StarDist2D
from util import *
# from skimage.util import random_noise

import matplotlib.pyplot as plt


#
# def add_noise(img):
#     # Adding Gaussian noise
#     noise_img = random_noise(img, mode='s&p', amount=0.3)
#     noise_img = np.array(255 * noise_img, dtype='uint8')
#     return noise_img


def histogram(img):
    hist = cv2.calcHist(img, [0], None, [256], [0, 256])
    # plt.imshow(hist)
    # plt.show()
    plt.hist(hist.ravel(), 256, [0, 256])
    plt.show()


# def gaussian_filter(img):
#     img = add_noise(img)
#     gaussian = cv2.GaussianBlur(img, (5, 5), 0)
#     show_2_images(img, gaussian, "Input Image", "Gaussian Filter")
#
#
# def median_blur(img):
#     img = add_noise(img)
#     median = cv2.medianBlur(img, 5)
#     show_2_images(img, median, "Input Image", "Median Filter")
#
#
# def bilateral(img):
#     img = add_noise(img)
#     blur = cv2.bilateralFilter(img, 9, 75, 75)
#     show_2_images(img, blur, "Input Image", "Bilateral Filter")


def canny_edge(img):
    canny_edge_image = cv2.Canny(img, 100, 200)
    show_2_images(img, canny_edge_image, "Input Image", "Edge Image")

    return canny_edge_image


def threshold(img):
    ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return thresh
    # show_image_cv2(thresh)


def segmentation_morphology(image, thresh):
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    show_multiple_images(image, opening, sure_bg, sure_fg, unknown, 'Original', 'Opening', "sure_bg", "sure_fg",
                         "The difference")


def contours_of_image(image, image_gray):
    thresh = threshold(image_gray)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Draw all contours from the image
    # The number -1 signifies drawing all contours
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    show_image(image)


if __name__ == "__main__":
    path_of_image = '/home/renos/Desktop/GitHub Projects/e-sante/e-sante/100119_d5_front.png'

    shape_image = '/home/renos/Downloads/star_shape.png'

    image = read_image(path=shape_image)
    show_image(image)
    image_gray = convert_to_gray(image=image)
    contours_of_image(image=image, image_gray=image_gray)

    # segmentation_morphology(image=image, thresh=threshold(img=image))
