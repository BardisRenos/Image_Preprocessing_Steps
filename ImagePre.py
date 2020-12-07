import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_image(img):
    cv2.imshow("Given Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def read_image(path):
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img_gray


def show_multiple_images(img, img_2, title_1, title_2):
    plt.figure(figsize=(20, 12))
    plt.subplot(121), plt.imshow(img, cmap='gray'), plt.title(title_1)
    plt.subplot(122), plt.imshow(img_2, cmap='gray'), plt.title(title_2)
    plt.show()


def add_noise(img):
    from skimage.util import random_noise
    # Adding Gaussian noise
    noise_img = random_noise(img, mode='s&p', amount=0.3)
    noise_img = np.array(255 * noise_img, dtype='uint8')
    return noise_img

def histogram(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.imshow(hist)
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

# def segmentation_morphology(img):


# img_gray = cv2.resize(img_gray, (350, 350), interpolation=cv2.INTER_AREA)
# print(img_gray.shape)

# ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

if __name__ == "__main__":
    path_of_image = '/home/renos/Desktop/e-sante/e-sante/100128_d1_front.png'
    histogram(read_image(path_of_image))
