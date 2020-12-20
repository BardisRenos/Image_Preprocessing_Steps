# Test 1
# 1. Take cell cluster image (png)
# 2. Detect each cell and count the total number of cells
# 3. Explain the approach you took, and the pros and cons compared to other potential
# approached

import cv2
import numpy as np


def task1():
    img = cv2.imread('/home/renos/Desktop/Cell cluster (1).png')
    H, W = img.shape[:2]
    original = img.copy()
    original = original[10:H, 0:W - 100]
    hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)

    hsv_lower = np.array([0, 0, 0])
    hsv_upper = np.array([180, 180, 180])
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)

    cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    count = 0
    for c in cnts:
        if 35 < cv2.contourArea(c):
            # print("The area is: ", cv2.contourArea(c))
            count += 1
            cv2.drawContours(original, [c], -1, (0, 255, 0), 3)

    print(count)
    cv2.imshow('original', original)
    cv2.waitKey()


# This is another approach but did not work quite well.

# def task1_1():
#     img = cv2.imread('/home/renos/Desktop/Cell cluster (1).png')
#     H, W = img.shape[:2]
#     original = img.copy()
#     original = img[10:H, 0:W - 100]
#     hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
#     blur = cv2.blur(original, (5, 5))
#     img_gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
#     thresh = cv2.threshold(img_gray, 155, 220, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#     D = ndimage.distance_transform_edt(thresh)
#     localMax = peak_local_max(D, indices=False, min_distance=20,
#                               labels=thresh)
#
#     markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
#     labels = watershed(-D, markers, mask=thresh)
#     print("[Information] {} unique segments found".format(len(np.unique(labels)) - 1))
#
#     for label in np.unique(labels):
#         # if the label is zero, we are examining the 'background'
#         # so simply ignore it
#         if label == 0:
#             continue
#         # otherwise, allocate memory for the label region and draw
#         # it on the mask
#         mask = np.zeros(img_gray.shape, dtype="uint8")
#         mask[labels == label] = 255
#         # detect contours in the mask and grab the largest one
#         cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
#                                 cv2.CHAIN_APPROX_SIMPLE)
#         cnts = imutils.grab_contours(cnts)
#         # c = max(cnts, key=cv2.contourArea)
#         cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)
#
#     cv2.imshow("Image", original)
#     cv2.waitKey(0)


def task2(path_file):
    import os
    import natsort
    import matplotlib.pyplot as plt

    def sort_images(path_file):
        dir_files = os.listdir(path=path_file)
        list_of_files = natsort.natsorted(dir_files, reverse=False)
        image_list = []
        for filename in list_of_files:
            if "ch02" in filename:
                image_list.append(cv2.imread(path_file + "/" + filename))
        return image_list

    def create_z_stack():
        list_of_files = sort_images(path_file)
        z_stack_image = np.concatenate(list_of_files, axis=2)
        print(z_stack_image.shape)
        return z_stack_image

    def show_single_image(img):
        cv2.imshow("Image", img)
        cv2.waitKey()

    def show_3D_images():
        from numpy import load
        arr = create_z_stack()
        fig, ax = plt.figure()
        a = arr[:, :, 1]
        ax.imshow(a, cmap='gray')
        ax.axis('off')
        plt.show()

    # show_3D_images()

    def sort_contours(contour_list):
        list_contours = []
        for i in contour_list:
            if cv2.contourArea(i) > 0:
                list_contours.append(i)
        return sorted(list_contours)

    def remove_circles():
        images = sort_images(path_file)
        images = [cv2.resize(images[i], (1000, 900)) for i in range(0, len(images))]
        image = images[0].copy()
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        mask = cv2.threshold(image_gray, 10, 255, cv2.THRESH_BINARY)[1]
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        # cv2.drawContours(image, [contours], -1, (0, 255, 0), 3)
        stencil = np.zeros(image.shape).astype(image.dtype)

        list_contours = sort_contours(contours)
        print(max(list_contours))
        for c in contours:
            if cv2.contourArea(c) == max(list_contours):
                cv2.drawContours(image, [c], -1, (0, 255, 0), 3)
                # # sort contours by largest first (if there are more than one)
                # result = 255 - mask
                # show_single_image(image)

        # show_single_image(cropped)

    # For the tasks 4 and 5 I will create the main body of preprocessing and after I created two methods
    # to calculate the z position of each cell and the mean z position of all cells

    def image_preprocessing():
        images = sort_images(path_file)
        images = [cv2.resize(images[i], (1000, 900)) for i in range(0, len(images))]
        image = images[0].copy()
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sigma = 2
        main_img = cv2.GaussianBlur(gray_image, (0, 0), sigma, 0)
        ret, threshold_img = cv2.threshold(main_img, 10, 255, cv2.THRESH_BINARY)
        threshold = cv2.bitwise_not(threshold_img)
        cnts = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        return cnts, image

    def z_position():
        cnts, image = image_preprocessing()
        for c in cnts:
            if 2 < cv2.contourArea(c) < 150:
                x, y, w, h = cv2.boundingRect(c)
                print("The position is on :", x, y)
                cv2.drawContours(image, [c], -1, (0, 255, 0), 3)
        show_single_image(image)

    # z_position()

    def calculate_mean_z_position():
        cnts, image = image_preprocessing()
        for c in cnts:
            M = cv2.moments(c)
            try:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            except ZeroDivisionError:
                print("Division with zero is not allowed")
            print(M["m10"])
            print(cX, cY)

    # calculate_mean_z_position()


if __name__ == '__main__':
    # task1()
    task2('/home/renos/Desktop/New 5x merged')
