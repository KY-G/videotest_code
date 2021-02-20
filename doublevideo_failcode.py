'''
简单的双进程展示双摄像头图像
按 s 结束

运行几秒后出现卡顿问题，现选择双进程进行
'''

import cv2
import numpy as np
import tkinter as tk

cap = cv2.VideoCapture(1)      # 参数为设备索引号，0代表内置摄像头
cap_2 = cv2.VideoCapture(0)

if cap.isOpened():
    if cap_2.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'));
        cap_2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'));
#设置显示界面宽高
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap_2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap_2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while(True):
    ret, frame = cap.read()     # 返回一个布尔值(ret)
    ret_2, frame_2 = cap_2.read()
    # if not ret or not ret_2:
    #     print("摄像头读取错误")
    #     break
    frame = cv2.flip(frame, 1)  # 若无本语句，摄像头非镜像
    frame_2 = cv2.flip(frame_2, 1)
    cv2.imshow("video", frame)
    cv2.imshow("video_2", frame_2)
    c = cv2.waitKey(50)
    if cv2.waitKey(1) == ord("s"):  # 按's' 退出，也可以设置其他键，用ord()转换为ASCIIwaitKey()返回ASCII码
        break

cap.release()
cap_2.release()
cv2.destroyAllWindows()