import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_image(img):
    plt.imshow(img, cmap='gray')
    plt.title("Given Image")
    plt.show()
    # cv2.imshow("Given Image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def read_image(path):
    img = cv2.imread(path)
    return img


def convert_to_gray(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return img_gray


def resize_image(image, w, h):
    resized_img = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    print(f"The image from {image.shape[:2]} to {resized_img.shape[:2]}")


def show_2_images(img, img_2, title_1, title_2):
    plt.figure(figsize=(20, 12))
    plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title(title_1)
    plt.subplot(122), plt.imshow(img_2, cmap='gray'), plt.title(title_2)
    plt.show()


def show_multiple_images(img, img_2, img_3, img_4, img_5, title_1, title_2, title_3, title_4, title_5):
    plt.figure(figsize=(20, 12))
    plt.subplot(321), plt.imshow(img, cmap='gray'), plt.title(title_1)
    plt.subplot(322), plt.imshow(img_2, cmap='gray'), plt.title(title_2)
    plt.subplot(323), plt.imshow(img_3, cmap='gray'), plt.title(title_3)
    plt.subplot(324), plt.imshow(img_4, cmap='gray'), plt.title(title_4)
    plt.subplot(325), plt.imshow(img_5, cmap='gray'), plt.title(title_5)
    plt.show()


def add_noise(img):
    from skimage.util import random_noise
    # Adding Gaussian noise
    noise_img = random_noise(img, mode='s&p', amount=0.3)
    noise_img = np.array(255 * noise_img, dtype='uint8')
    return noise_img


def histogram(img):
    hist = cv2.calcHist(img, [0], None, [256], [0, 256])
    # plt.imshow(hist)
    # plt.show()
    plt.hist(hist.ravel(), 256, [0, 256])
    plt.show()


def gaussian_filter(img):
    img = add_noise(img)
    gaussian = cv2.GaussianBlur(img, (5, 5), 0)
    show_multiple_images(img, gaussian, "Input Image", "Gaussian Filter")


def median_blur(img):
    img = add_noise(img)
    median = cv2.medianBlur(img, 5)
    show_multiple_images(img, median, "Input Image", "Median Filter")


def bilateral(img):
    img = add_noise(img)
    blur = cv2.bilateralFilter(img, 9, 75, 75)
    show_multiple_images(img, blur, "Input Image", "Bilateral Filter")


def canny_edge(img):
    canny_edge = cv2.Canny(img, 100, 200)
    show_multiple_images(img, canny_edge, "Input Image", "Edge Image")


def threshold(img):
    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    return thresh
    # show_image(thresh)


def segmentation_morphology(image, thresh):

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    show_multiple_images(image, opening, sure_bg, sure_fg, unknown, 'Original', 'Opening', "sure_bg", "sure_fg", "The "
                                                                                                                 "difference")


if __name__ == "__main__":
    path_of_image = '/home/renos/Desktop/e-sante/e-sante/100128_d1_front.png'
    segmentation_morphology(read_image(path_of_image), threshold(convert_to_gray(read_image(path_of_image))))

