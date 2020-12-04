import cv2
import numpy


def show_image(img):
    cv2.imshow("Given Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread('/home/eAdmin/Desktop/Images/100128_d6_front.png')
print(img.shape)


# if __name__ == "__main__":
#     show_image(img)