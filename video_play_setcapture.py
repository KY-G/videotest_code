'''
瞳孔直径检测演示程序，单目
每隔1s发送一个平均瞳孔值
可拖动调节二值化阈值
加入亮度、对比度拖动调节
按 s 结束运行

计划进行分辨率调节，以实现完全显示迈德威视摄像头图像
分辨率设置最耗时间660ms
第二位的是摄像头图片读取，图片处理到时不太耗时间
紧接着就是压缩图片输出  20ms左右
分辨率设置不能加载循环里面，还是要调用SDK，一次性设置完毕，只进行图像处理应该会很省时间

已验证无驱动摄像头，OPENCV的分辨率设置可以直接加在循环外，可以将大图像处理，然后压缩显示，时间足够

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
    height_vid, width_vid = dst.shape[:2]
    frame_1 = cv2.resize(dst, (int(width_vid *compression_W ), int(height_vid *compression_H)))
    cv2.imshow("video", frame_1)
#    cv2.imshow("video", frame)
    return dst

cap = cv2.VideoCapture(0)      # 参数为设备索引号，0代表内置摄像头
#cap_2 = cv2.VideoCapture(1)
t_start = cv2.getTickCount()
j = 0
w0 = 0
h0 = 0
w_ave = 0
h_ave = 0
compression_H = 1
compression_W = 1

cv2.namedWindow('Color_con_bri Track Bar')
# hh = 'Adj_gray'
# #hl = 'Min'
# wnd = 'Colorbars'
cv2.createTrackbar("Adj_gray", "Color_con_bri Track Bar", 0, 255, trackChaned)
cv2.createTrackbar("Adj_contrast", "Color_con_bri Track Bar", 0, 20, trackChaned)
cv2.createTrackbar("Adj_brightness", "Color_con_bri Track Bar", 0, 1000, trackChaned)

# 设置默认值
cv2.setTrackbarPos('Adj_gray', 'Color_con_bri Track Bar', 9)
cv2.setTrackbarPos('Adj_contrast', 'Color_con_bri Track Bar', 9)
cv2.setTrackbarPos('Adj_brightness', 'Color_con_bri Track Bar', 15)

# #
# if cap.isOpened():
#     if cap_2.isOpened():
# #设置显示界面宽高
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#         cap_2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#         cap_2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
t_setTrackbar = cv2.getTickCount()
while(True):
    ret, frame = cap.read()     # 返回一个布尔值(ret)
    t_read = cv2.getTickCount()
    cap.set(3, 1000)#宽
    cap.set(4, 750)#高
    t_imgset = cv2.getTickCount()
    frame = cv2.flip(frame, 1)  # 若无本语句，摄像头非镜像
    c = cv2.waitKey(50)

#亮度bla、对比度con调节

    t_waiting = cv2.getTickCount()
    gray = cv2.cvtColor(Contrast_and_Brightness(frame), cv2.COLOR_BGR2GRAY)
    t_gray_put = cv2.getTickCount()
#    t_gray = cv2.getTickCount()
    Adj = cv2.getTrackbarPos("Adj_gray", "Color_con_bri Track Bar")

    ret, binary = cv2.threshold(gray, Adj, 255, cv2.THRESH_BINARY)
    t_binary = cv2.getTickCount()
    height_bin, width_bin = binary.shape[:2]
    binary_1 = cv2.resize(binary, (int(width_bin *compression_W), int(height_bin *compression_H)))
    cv2.imshow("binary", binary_1)
    t_binary_put = cv2.getTickCount()
    closing = binary
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    t_morphologyEx = cv2.getTickCount()
    img3, contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    t_find = cv2.getTickCount()
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
                    t_find_eyes = cv2.getTickCount()
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
                #t_text = cv2.getTickCount()

                # 图像，文字内容， 坐标 ，字体，大小，颜色，字体厚度


    # cv2.imshow("gray", gray)
    # cv2.imshow("binary", binary)
    height_clo, width_clo = binary.shape[:2]
    closing_1 = cv2.resize(binary, (int(width_clo *compression_W), int(height_clo *compression_H)))
    cv2.imshow('closing_1', closing_1)
    t_closing_put = cv2.getTickCount()
#    t_closing = cv2.getTickCount()

    time_setTrackbar = (t_setTrackbar - t_start) / cv2.getTickFrequency()
    time_read = (t_read - t_setTrackbar) / cv2.getTickFrequency()
    time_imgset = (t_imgset - t_read) / cv2.getTickFrequency()
    time_waiting = (t_waiting - t_imgset) / cv2.getTickFrequency()
    time_gray_put = (t_gray_put - t_waiting) / cv2.getTickFrequency()
    time_binary = (t_binary - t_gray_put) / cv2.getTickFrequency()
    time_binary_put = (t_binary_put - t_binary) / cv2.getTickFrequency()
    time_morphologyEx = (t_morphologyEx - t_binary_put) / cv2.getTickFrequency()
    time_find = (t_find - t_morphologyEx) / cv2.getTickFrequency()
    time_closing_put = (t_closing_put - t_find) / cv2.getTickFrequency()

    print("time_setTrackbar : %s ms" % (time_setTrackbar * 1000))   #163.378ms
    print("time_read  : %s ms" % (time_read * 1000))  #67.332ms
    print("time_imgset  : %s ms" % (time_imgset * 1000))#660ms
    print("time_waiting  : %s ms" % (time_waiting * 1000))#51.75ms
    print("time_binary  : %s ms" % (time_binary * 1000))#14ms
    print("time_binary_put  : %s ms" % (time_binary_put * 1000))#23
    print("time_morphologyEx  : %s ms" % (time_morphologyEx * 1000))#1ms
    print("time_drawCircle  : %s ms" % (time_find * 1000))#0.28ms
    print("time_closing_put  : %s ms" % (time_closing_put * 1000))#23ms

    # time_setTrackbar = (t_setTrackbar - t_start) / cv2.getTickFrequency()
    # time_read = (t_read - t_setTrackbar) / cv2.getTickFrequency()
    # time_waiting = (t_waiting - t_read) / cv2.getTickFrequency()
    # time_gray = (t_gray - t_waiting) / cv2.getTickFrequency()
    # time_binary = (t_binary - t_gray) / cv2.getTickFrequency()
    # time_morphologyEx = (t_morphologyEx - t_binary) / cv2.getTickFrequency()
    # time_find = (t_find - t_morphologyEx) / cv2.getTickFrequency()
    # time_closing = (t_closing - t_find) / cv2.getTickFrequency()
    #
    # print("time_setTrackbar : %s ms" % (time_setTrackbar * 1000))  # 226.6ms
    # print("time_read  : %s ms" % (time_read * 1000))  # 5ms
    # print("time_waiting  : %s ms" % (time_waiting * 1000))  # 51ms
    # print("time_gray  : %s ms" % (time_gray * 1000))  # 3ms
    # print("time_binary  : %s ms" % (time_binary * 1000))  # 0.34ms
    # print("time_morphologyEx  : %s ms" % (time_morphologyEx * 1000))  # 1.04ms
    # print("time_find  : %s ms" % (time_find * 1000))  # 0.25ms
    # print("time_closing  : %s ms" % (time_closing * 1000))  # 5ms

    if cv2.waitKey(1) == ord("s"):  # 按's' 退出，也可以设置其他键，用ord()转换为ASCIIwaitKey()返回ASCII码
        break

'''
加上像素重新设置处理和压缩的情况下一个周期900ms
分辨率设置最耗时间660ms
第二位的是摄像头图片读取，图片处理到时不太耗时间
紧接着就是压缩图片输出  20ms左右

不加像素重新设置处理的情况下一个循环大概340ms（？？）
其他图像操作基本上10ms以内都可以处理完。加上延时50ms大概60ms一个周期
'''

cap.release()
cv2.destroyAllWindows()
