import cv2
import imutils
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from matplotlib import pyplot as plt
import os


# This is a (demo) Test regarding the image manipulation and applying filters.
# In order to produce a valid algorithm to detect correctly. In the function image_manipulation
# are all the necessary steps and the parameters.
# Also, in this demo exists the plotting and saving mode.

class ImageDetection(object):
    pass

    @staticmethod
    def save_image(img: str, name: str):
        path_to_save = '/home/eAdmin/Desktop/oculus/oculus_evaluation/oculus_evaluation/Unit_Tests/Image_results'
        cv2.imwrite(os.path.join(path_to_save, name + '.jpg'), img)

    @staticmethod
    def show_images_stages(img, img_with_circle, thresh, morph, mask, image_with_contours):

        plt.figure(figsize=(20, 12))

        plt.subplot(231), plt.imshow(img, cmap='gray'), plt.title('Image')
        plt.subplot(232), plt.imshow(img_with_circle, cmap='gray'), plt.title('Image with circle')
        plt.subplot(233), plt.imshow(thresh, cmap='gray'), plt.title('1st Threshold')

        plt.subplot(234), plt.imshow(morph, cmap='gray'), plt.title('Morph')
        plt.subplot(235), plt.imshow(mask, cmap='gray'), plt.title('Mask')
        plt.subplot(236), plt.imshow(image_with_contours, cmap='gray'), plt.title('Final Image')
        plt.show()

    @staticmethod
    def show_image(given_image):
        # Show an image (simple) OR using for showing an image after the contours
        cv2.imshow("Image", given_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def image_manipulation(image_path_file: str):
        # Read image
        # print(image_from_path)
        image = cv2.imread(image_path_file)
        image_raw = image.copy()
        image_to_show = image.copy()
        h, w = image_raw.shape[:2]
        cv2.circle(image_raw, ((w // 2) + 1, (h // 2) - 5), 622, (255, 255, 255), 350)
        # threshold on white
        # Define lower and uppper limits
        lower = np.array([220, 220, 220])
        upper = np.array([255, 255, 255])
        # Create mask to only select black
        thresh = cv2.inRange(image_raw, lower, upper)
        # apply morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        # invert morp image
        mask = 255 - morph
        # apply mask to image
        result = cv2.bitwise_and(image_raw, image_raw, mask=mask)

        img_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        ret, th1 = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY)
        D = ndimage.distance_transform_edt(th1)
        localMax = peak_local_max(D, indices=False, min_distance=20, labels=mask)
        markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        labels = watershed(-D, markers, mask=mask)
        print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
        contours = []

        for label in np.unique(labels):
            if label == 0:
                continue

            mask = np.zeros(img_gray.shape, dtype="uint8")
            mask[labels == label] = 255
            # detect contours in the mask and grab the largest one
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            # c = max(cnts, key=cv2.contourArea)
            # hull = cv2.convexHull(c)
            cv2.drawContours(image, cnts, -1, (0, 255, 0), 2)

        # Show a single image
        # ImageDetection.show_image(morph)

        # Save the image to the directory to see the results after applying the contours on the pills.
        # You can uncomment the command and save your results on your local folder
        # ImageDetection.save_image(image, "res_8")

        # Show a multiple plot when an image pass the pre processing stages
        ImageDetection.show_images_stages(img=image_to_show, img_with_circle=image_raw, thresh=thresh, morph=morph,
                                          mask=result, image_with_contours=image)


if __name__ == '__main__':
    image_from_path = '/home/eAdmin/Desktop/oculus/oculus_evaluation/oculus_evaluation/Unit_Tests/Images' \
                      '/100100_d2_front.png'
    ImageDetection.image_manipulation(image_from_path)
