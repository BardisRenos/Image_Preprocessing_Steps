import cv2
import numpy as np


def show_image(img):
    cv2.imshow("Given Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def add_noise(img):
    from skimage.util import random_noise
    # Adding Gaussian noise
    noise_img = random_noise(img, mode='s&p', amount=0.3)
    noise_img = np.array(255*noise_img, dtype='uint8')
    show_image(noise_img)


img = cv2.imread('/home/renos/Desktop/e-sante/e-sante/100128_d1_front.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# img_gray = cv2.resize(img_gray, (350, 350), interpolation=cv2.INTER_AREA)
# print(img_gray.shape)

# ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

if __name__ == "__main__":
    add_noise(img_gray)
