'''
瞳孔直径检测演示程序，单目
每隔1s发送一个平均瞳孔值
可拖动调节二值化阈值、亮度、对比度
按 s 结束运行
'''
import cv2
import numpy as np
import tkinter as tk

def trackChaned(x):
  pass

def Contrast_and_Brightness(img):

    con = cv2.getTrackbarPos("Adj_contrast", "Color_con_bri Track Bar")*0.1
    bri = cv2.getTrackbarPos("Adj_brightness", "Color_con_bri Track Bar")*0.1

    blank = np.zeros(img.shape, img.dtype)
    dst = cv2.addWeighted(img, con, blank, 1-con, bri)
    cv2.imshow("video", dst)
    return dst

cap = cv2.VideoCapture(0)      # 参数为设备索引号，0代表内置摄像头
#cap_2 = cv2.VideoCapture(1)

j = 0
w0 = 0
h0 = 0
w_ave = 0
h_ave = 0


cv2.namedWindow('Color_con_bri Track Bar')
# hh = 'Adj_gray'
# #hl = 'Min'
# wnd = 'Colorbars'
cv2.createTrackbar("Adj_gray", "Color_con_bri Track Bar", 0, 255, trackChaned)
cv2.createTrackbar("Adj_contrast", "Color_con_bri Track Bar", 0, 20, trackChaned)
cv2.createTrackbar("Adj_brightness", "Color_con_bri Track Bar", 0, 1000, trackChaned)

# 设置默认值
cv2.setTrackbarPos('Adj_gray', 'Color_con_bri Track Bar', 59)
cv2.setTrackbarPos('Adj_contrast', 'Color_con_bri Track Bar', 9)
cv2.setTrackbarPos('Adj_brightness', 'Color_con_bri Track Bar', 15)

#
# if cap.isOpened():
#     if cap_2.isOpened():
# #设置显示界面宽高
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#         cap_2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#         cap_2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while(True):
    ret, frame = cap.read()     # 返回一个布尔值(ret)
    frame = cv2.flip(frame, 1)  # 若无本语句，摄像头非镜像
    c = cv2.waitKey(50)
#亮度bla、对比度con调节


    gray = cv2.cvtColor(Contrast_and_Brightness(frame), cv2.COLOR_BGR2GRAY)

    Adj = cv2.getTrackbarPos("Adj_gray", "Color_con_bri Track Bar")

    ret, binary = cv2.threshold(gray, Adj, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    img3, contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(0, len(contours)):
        cnt = contours[i]
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.rectangle(closing, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.drawContours(gray, contours, -1, (0, 255, 0), 3)
        cv2.drawContours(closing, contours, -1, (0, 0, 255), 1)
        size = closing.shape
        ww = size[1]  # 宽度
        hh = size[0]  # 高度

        xw = w * 0.1
        yh = h * 0.1

        if xw >= 1 and yh <= 30 and xw < 2*yh < 3*xw:
            if x > ww / 20 and y > hh / 20:
                j = j + 1
                w0 = (w + w0)/2
                h0 = (h + h0)/2


                if j >= 20:
                    w_ave = w0 * 0.1
                    h_ave = h0 * 0.1
                    print('w = %.4f' % w_ave, 'mm')
                    print('h = %.4f' % h_ave, 'mm')

                    print(type(contours))
                    print(len(contours))
                    j = 0

                font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体,读取到多个图形时会导致文字重叠
                ss = 'W = ' + str(format(xw, '.4f')) + 'mm'
                W_ave1s = 'Wave_1s = ' + str(format(w_ave, '.4f')) + 'mm'
                cv2.putText(closing, ss, (int(ww * 0.2), 50), font, 0.8, (0, 255, 0), 2)
                cv2.putText(closing, W_ave1s, (10, 420), font, 0.8, (0, 255, 0), 2)
      #          cv2.putText(frame, ss, (int(ww * 0.2), 50), font, 0.8, (0, 255, 0), 2)


                # 图像，文字内容， 坐标 ，字体，大小，颜色，字体厚度
                ss = 'H = ' + str(format(yh, '.4f')) + 'mm'
                H_ave1s = 'Have_1s = ' + str(format(h_ave, '.4f')) + 'mm'
                cv2.putText(closing, ss, (int(ww * 0.6), 50), font, 0.8, (0, 255, 0), 2)
                cv2.putText(closing, H_ave1s, (10, 460), font, 0.8, (0, 255, 0), 2)
    #            cv2.putText(frame, ss, (int(ww * 0.6), 50), font, 0.8, (0, 255, 0), 2)

                # 图像，文字内容， 坐标 ，字体，大小，颜色，字体厚度




    cv2.imshow("gray", gray)
    cv2.imshow("binary", binary)
    cv2.imshow('closing', closing)
    if cv2.waitKey(1) == ord("s"):  # 按's' 退出，也可以设置其他键，用ord()转换为ASCIIwaitKey()返回ASCII码
        break

cap.release()
cv2.destroyAllWindows()

#
# ret, binary = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)
# # ret, binary = cv2.threshold(gray,15,255,cv2.THRESH_BINARY)
#
# # 2. cv2.MORPH_OPEN 先进行腐蚀操作，再进行膨胀操作
# # kernel = np.ones((5, 5), np.uint8)
# # opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
# # cv2.imshow('opening', opening)
#
# # 3. cv2.MORPH_CLOSE 先进行膨胀，再进行腐蚀操作
# kernel = np.ones((5, 5), np.uint8)
# closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
# # cv2.imshow('closing', closing)
#
# # binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, anchor=(2, 0), iterations=5)
# # binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k1)
#
# img3, contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# # contours = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# # contours,hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# # contours, hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NOME)
#
# # cv2.imshow('closing', closing)
#
# # BGR
# for i in range(0, len(contours)):
#     cnt = contours[i]
#     x, y, w, h = cv2.boundingRect(cnt)
#     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
#     # cv2.rectangle(binary,(x,y),(x+w,y+h),(0,255,0),1)
#     # cv2.rectangle(opening,(x,y),(x+w,y+h),(0,255,0),1)
#     cv2.rectangle(closing, (x, y), (x + w, y + h), (0, 255, 0), 1)
#
#     cv2.drawContours(img, contours, -1, (0, 0, 255), 1)
#     # cv2.drawContours(binary,contours,-1,(0,0,255),3)
#     # cv2.drawContours(opening,contours,-1,(0,0,255),3)
#     cv2.drawContours(closing, contours, -1, (0, 0, 255), 1)
#
#     size = closing.shape
#     ww = size[1]  # 宽度
#     hh = size[0]  # 高度
#
#     xw = w * 0.05
#     yh = h * 0.05
#
#     if xw >= 1 and yh <= 15:
#         if x > ww / 3 and y > hh / 4:
#             print('w =', w * 0.0499, 'mm')
#             print('h =', h * 0.0499, 'mm')
#
#             font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
#             ss = 'W = ' + str(xw) + 'mm'
#             imgzi = cv2.putText(closing, ss, (int(ww * 0.2), 50), font, 0.8, (0, 255, 0), 2)
#             imgzi = cv2.putText(img, ss, (int(ww * 0.2), 50), font, 0.8, (0, 255, 0), 2)
#             # 图像，文字内容， 坐标 ，字体，大小，颜色，字体厚度
#             ss = 'H = ' + str(yh) + 'mm'
#             imgzi = cv2.putText(closing, ss, (int(ww * 0.6), 50), font, 0.8, (0, 255, 0), 2)
#             imgzi = cv2.putText(img, ss, (int(ww * 0.6), 50), font, 0.8, (0, 255, 0), 2)
#             # 图像，文字内容， 坐标 ，字体，大小，颜色，字体厚度
#
# print(type(contours))
# print(len(contours))

# cv2.imshow("binary", binary)
# # cv2.imshow('opening', opening)
# cv2.imshow('closing', closing)
# cv2.waitKey(0)
