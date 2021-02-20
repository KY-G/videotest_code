import cv2
import numpy as np
from matplotlib import pyplot as plt
#直方图计算

def plot_demo(image):
    plt.hist(image.ravel(), 256, [0, 256])
    plt.show("直方图")

def image_hist(img):
    color = ('blue', 'green', 'red')
    for i, color in enumerate(color):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.show()

img_lig1 = cv2.imread("eye_yuan.png")
# cv2.namedWindow("img_lig1", cv2.WINDOW_AUTOSIZE)
# cv2.imshow("img_lig1", img_lig1)
# plot_demo(img_lig1)
# img_lig2 = cv2.imread("hongwai_gray_light2.png")
# img_hwlig1 = cv2.imread("hongwai_gray_hwlight.png")
# img_hwlig2 = cv2.imread("hongwai_gray_hwlight2.png")
# img_eyes_hwlig2 = cv2.imread("eyes_hongwai_hwlig.png")

cv2.namedWindow("img_lig1", cv2.WINDOW_AUTOSIZE)
cv2.imshow("img_lig1", img_lig1)
#plot_demo(img_lig1)
image_hist(img_lig1)

#
# cv2.namedWindow("img_lig2", cv2.WINDOW_AUTOSIZE)
# cv2.imshow("img_lig2", img_lig2)
# #plot_demo(img_lig2)
# image_hist(img_lig2)
#
#
# cv2.namedWindow("img_hwlig1", cv2.WINDOW_AUTOSIZE)
# cv2.imshow("img_hwlig1", img_hwlig1)
# #plot_demo(img_hwlig1)
# image_hist(img_hwlig1)
#
# cv2.namedWindow("img_hwlig2", cv2.WINDOW_AUTOSIZE)
# cv2.imshow("img_hwlig2", img_hwlig2)
# #plot_demo(img_hwlig2)
# image_hist(img_hwlig2)
#
# cv2.namedWindow("img_eyes_hwlig2", cv2.WINDOW_AUTOSIZE)
# cv2.imshow("img_eyes_hwlig2", img_eyes_hwlig2)
# #plot_demo(img_hwlig2)
# image_hist(img_eyes_hwlig2)

cv2.waitKey(0)

cv2.destroyAllWindows()
