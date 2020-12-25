import cv2
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
