'''
瞳孔直径检测程序，单目
每隔1s发送一个平均瞳孔值
加入白光检测，填充光点，受环境干扰较大，容易出现瞳孔像素点散乱。
按 s 结束运行
'''

import cv2
import numpy as np

def trackChaned(x):
  pass



cap = cv2.VideoCapture(0)      # 参数为设备索引号，0代表内置摄像头
j = 0
w0 = 0
h0 = 0
w_ave = 0
h_ave = 0


cv2.namedWindow('Color Track Bar')
hh = 'Adj_gray'
#hl = 'Min'
wnd = 'Colorbars'
cv2.createTrackbar("Adj_gray", "Color Track Bar", 0, 255, trackChaned)
#cv2.createTrackbar("Min", "Color Track Bar", 0, 255, trackChaned)
# 设置默认值
cv2.setTrackbarPos('Adj_gray', 'Color Track Bar', 20)

while(True):
    ret, frame = cap.read()     # 返回一个布尔值(ret)
    frame = cv2.flip(frame, 1)  # 若无本语句，摄像头非镜像
    cv2.namedWindow("video", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("video", frame)
    c = cv2.waitKey(50)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # 色彩空间转换为hsv，分离.
    low = np.array([0, 0, 221])
    high = np.array([180, 30, 255])

    split = cv2.inRange(src=hsv, lowerb=low, upperb=high)  # HSV高低阈值，提取图像部分区域
    # 寻找白色的像素点坐标。
    # 白色像素值是255，所以np.where(dst==255)
    xy = np.column_stack(np.where(split == 255))
    #print(xy)

    # 在原图的红色数字上用 黑色 描点填充。
    for c in xy:
       # print(c)
        # 注意颜色值是(b,g,r)，不是(r,g,b)
        # 坐标:c[1]是x,c[0]是y
        cv2.circle(img=frame, center=(int(c[1]), int(c[0])), radius=1, color=(0, 0, 0), thickness=1)
    cv2.imshow("split_video", frame)
    #亮度bla、对比度c调节
    con = 0.9
    bla = 1.5
    h, w, ch = frame.shape
    blank = np.zeros([h, w, ch], frame.dtype)
    dst = cv2.addWeighted(frame, con, blank, 1 - con, bla)

    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    # # 利用cv2.minMaxLoc寻找到图像中最亮和最暗的点
    # (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    # # 在图像中提取ROI区域
    # ROI_gray = src[]

    Adj = cv2.getTrackbarPos("Adj_gray", "Color Track Bar")
    #  huh = cv2.getTrackbarPos("Min", "Color Track Bar")

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

        xw = w * 0.0499
        yh = h * 0.0499

        if xw >= 1 and yh <= 15 and xw < 2*yh < 3*xw:
            if x > ww / 5 and y > hh / 5:
                j = j + 1
                w0 = (w + w0)/2
                h0 = (h + h0)/2


                if j >= 20:
                    w_ave = w0 * 0.0499
                    h_ave = h0 * 0.0499
                    print('w = %.4f' % w_ave, 'mm')
                    print('h = %.4f' % h_ave, 'mm')

                    print(type(contours))
                    print(len(contours))
                    j = 0

                font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
                ss = 'W = ' + str(format(xw, '.4f')) + 'mm'
                W_ave1s = 'Wave_1s = ' + str(format(w_ave, '.4f')) + 'mm'
                imgzi = cv2.putText(closing, ss, (int(ww * 0.2), 50), font, 0.8, (0, 255, 0), 2)
                imgzi = cv2.putText(closing, W_ave1s, (10, 420), font, 0.8, (0, 255, 0), 2)
                imgzi = cv2.putText(frame, ss, (int(ww * 0.2), 50), font, 0.8, (0, 255, 0), 2)


                # 图像，文字内容， 坐标 ，字体，大小，颜色，字体厚度
                ss = 'H = ' + str(format(yh, '.4f')) + 'mm'
                H_ave1s = 'Have_1s = ' + str(format(h_ave, '.4f')) + 'mm'
                imgzi = cv2.putText(closing, ss, (int(ww * 0.6), 50), font, 0.8, (0, 255, 0), 2)
                imgzi = cv2.putText(closing, H_ave1s, (10, 460), font, 0.8, (0, 255, 0), 2)
                imgzi = cv2.putText(frame, ss, (int(ww * 0.6), 50), font, 0.8, (0, 255, 0), 2)

                # 图像，文字内容， 坐标 ，字体，大小，颜色，字体厚度




    cv2.imshow("gray", gray)
    cv2.imshow("binary", binary)
    cv2.imshow('closing', closing)
    if cv2.waitKey(1) == ord("s"):  # 按'e' 退出，也可以设置其他键，用ord()转换为ASCIIwaitKey()返回ASCII码
        break

cap.release()
cv2.destroyAllWindows()