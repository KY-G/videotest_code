'''
瞳孔直径检测演示程序，双目
功能已与单目的程序达到相同功能
按 s 结束一个进程运行
瞳孔直径显示已校准，3-5mm的瞳孔显示较准
将滑动调节和视频显示框做到一起显示

尚存问题：
当处理500万像素的图片时，仍旧会卡顿（现在处理200万像素的图片）
按 s 结束一个进程运行，无法同时结束两个
。。。。。。。。。。。。。。。。。。。2020.02.01。。。。。。。。。。。。。。。。。。。。
'''
import numpy as np
import cv2
import multiprocessing
from multiprocessing import Queue
import time
import imutils

def trackChaned(x):
  pass

def Contrast_and_Brightness(img):

    con = cv2.getTrackbarPos("Adj_contrast", 'Color Track Bar'+ str(id))*0.1
    bri = cv2.getTrackbarPos("Adj_brightness", 'Color Track Bar'+ str(id))*0.1

    blank = np.zeros(img.shape, img.dtype)
    dst = cv2.addWeighted(img, con, blank, 1-con, bri)
#    cv2.imshow("video", frame)
    return dst

def average(list):
    # "对列表的数值求和"
    s = 0
    for x in list:
        s += x
    if len(list) > 0:
        #"对列表数据求平均值"
        avg = s/len(list)*1 #调用sum函数求和
    else:
        avg = 0
    return avg


CAMERA_COUNT = 2  # 摄像头个数
#url = "rtsp://admin:12345678@192.168.128.45"  # 外置摄像头地址及密码
# 不同进程不能共享内存，定义队列进行进程通信
q = Queue()

def video_read(id):

    compression_H = 0.32
    compression_W = 0.32

    j = 0
    list_w = []
    list_h = []
    list_long = 20  # 求平均瞳孔直径时的采集值个数

    camera_id = id
    # 使用笔记本自带的摄像头
    if camera_id == 0:
        cap = cv2.VideoCapture(0)
    # 使用外置的摄像头
    if camera_id == 1:
        cap = cv2.VideoCapture(1)
#像素设置为500万，略有卡顿
    # cap.set(3, 2580)  # 宽
    # cap.set(4, 1940)  # 高

    cap.set(3, 2000)  # 宽
    cap.set(4, 1500)  # 高
    # 曝光度设置
    cap.set(15, -4)
    print("exposure={}".format(cap.get(15)))
    # 获取每一个视频的尺寸
    width = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(width, height)

    cv2.namedWindow('Color Track Bar'+ str(id))
    hh = 'Adj_gray'
    # hl = 'Min'
    wnd = 'Colorbars'
    cv2.createTrackbar("Adj_gray", 'Color Track Bar'+ str(id), 0, 255, trackChaned)
    cv2.createTrackbar("Adj_contrast", 'Color Track Bar'+ str(id), 0, 20, trackChaned)
    cv2.createTrackbar("Adj_brightness", 'Color Track Bar'+ str(id), 0, 1000, trackChaned)
    # 设置默认值
    cv2.setTrackbarPos('Adj_gray', 'Color Track Bar'+ str(id), 100)
    cv2.setTrackbarPos('Adj_contrast', 'Color Track Bar'+ str(id), 9)
    cv2.setTrackbarPos('Adj_brightness', 'Color Track Bar'+ str(id), 15)

    while (cap.isOpened()):
        #ret, frame = cap.read()

        q.put(cap.read()[1])  # 刚刚放进去一张图片
        if q.qsize() > 1:
            q.get()
        else:
            time.sleep(0.001)  # 然后马上取出来
            #p.start()

        frame = q.get()  # 等待队列放入图片，如果队列里面没有图片，那么它会「阻塞」在这里
        #frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_CUBIC)
        isEmpty = q.empty()
        if isEmpty == True:
#            print('队列中无数据！')
            time.sleep(0.010)
        else:
            print('队列中有图片！')
            # Frame = q.get(frame)
            time.sleep(0.020)

        frame = cv2.flip(frame, 1)  # 若无本语句，摄像头非镜像
        cv2.imshow('Color Track Bar' + str(id), imutils.resize(frame, width=640, height=480))

        con = cv2.getTrackbarPos("Adj_contrast", 'Color Track Bar' + str(id)) * 0.1
        bri = cv2.getTrackbarPos("Adj_brightness", 'Color Track Bar' + str(id)) * 0.1
        blank = np.zeros(frame.shape, frame.dtype)
        dst = cv2.addWeighted(frame, con, blank, 1 - con, bri)
        gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

        Adj = cv2.getTrackbarPos("Adj_gray", 'Color Track Bar'+ str(id))

        ret, binary = cv2.threshold(gray, Adj, 255, cv2.THRESH_BINARY)
        closing = binary
        kernel = np.ones((5, 5), np.uint8)
        closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        img3, contours, hierarchy = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # 第二个参数由cv2.RETR_TREE改为RETE_LIST,不分等级
        # 长度最好的状态为len(contours) = 2时，可以放宽到4？？
        if (len(contours) < 31):
            for i in range(0, len(contours)):
                cnt = contours[i]
                x, y, w, h = cv2.boundingRect(cnt)
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), int(1/compression_H))
                cv2.rectangle(closing, (x, y), (x + w, y + h), (0, 255, 0), int(1 / compression_H))
                # cv2.drawContours(gray, contours, -1, (0, 255, 0), int(1/compression_H))
                size = closing.shape
                photo_w = size[1]  # 宽度
                photo_h = size[0]  # 高度

                pupil_w = w * 0.0215
                pupil_h = h * 0.0215

                if (10 >= pupil_w >= 0.5 and 0.5 <= pupil_h <= 10) and (pupil_w < 2 * pupil_h < 3 * pupil_w):
                    if (photo_w * 0.7 > x > photo_w * 0.3) and (photo_h * 0.6 > y > photo_h * 0.3):
                        cv2.drawContours(closing, contours, -1, (0, 0, 255), int(1 / compression_H))
                        # 写入数组数据
                        if j < 20:
                            list_w.append(pupil_w)  # 从数组末端加入
                            list_h.append(pupil_h)
                            j = +1
                        else:
                            j = 0
                        # 删除数组首位元素
                        if len(list_w) == list_long + 1 | len(list_h) == list_long + 1:
                            del list_w[0]  # 删除数组首位
                            del list_h[0]

                        print('w = %.4f' % pupil_w, 'mm')  # 整数‘%d’
                        print('h = %.4f' % pupil_h, 'mm')
                        print(type(contours))
                        print(len(contours))

                        # print("Updated list_w : ", list_w) #显示数组所有的元素
                        w_ave = average(list_w)
                        h_ave = average(list_h)

                        font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体,读取到多个图形时会导致文字重叠
                        str_w = 'Width_pupil = ' + str(format(pupil_w, '.1f')) + 'mm'
                        str_W_ave1s = 'W_avg = ' + str(format(w_ave, '.1f')) + 'mm'
                        cv2.putText(closing, str_w, (int(photo_w * 0.01), int(photo_h * 0.1)), font,
                                    int(0.6 / compression_H + 1), (0, 255, 0), int(2 / compression_H + 1))
                        cv2.putText(closing, str_W_ave1s, (int(photo_w * 0.05), int(photo_h * 0.85)), font,
                                    int(0.8 / compression_H + 1), (0, 255, 0), int(2 / compression_H + 1))
                        # 文字在屏幕中的位置，
                        # 图像，文字内容， 坐标（横坐标，纵坐标） ，字体，大小，颜色，字体厚度

                        str_h = 'High_pupil = ' + str(format(pupil_h, '.1f')) + 'mm'
                        str_H_ave1s = 'H_avg = ' + str(format(h_ave, '.1f')) + 'mm'
                        cv2.putText(closing, str_h, (int(photo_w * 0.41), int(photo_h * 0.1)), font,
                                    int(0.6 / compression_H + 1), (0, 255, 0), int(2 / compression_H + 1))
                        cv2.putText(closing, str_H_ave1s, (int(photo_w * 0.05), int(photo_h * 0.95)), font,
                                    int(0.8 / compression_H + 1), (0, 255, 0), int(2 / compression_H + 1))
                        # 图像，文字内容， 坐标 ，字体（横坐标，纵坐标），大小，颜色，字体厚度
                else:
                    w_avg = average(list_w)
                    h_avg = average(list_h)
                    font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体,读取到多个图形时会导致文字重叠
                    str_W_ave1s = 'W_avg = ' + str(format(w_avg, '.1f')) + 'mm'
                    cv2.putText(closing, str_W_ave1s, (int(photo_w * 0.05), int(photo_h * 0.85)), font,
                                int(0.8 / compression_H + 1), (0, 255, 0), int(2 / compression_H + 1))
                    # 文字在屏幕中的位置，
                    # 图像，文字内容， 坐标（--横坐标，纵坐标） ，字体，大小，颜色，字体厚度

                    str_H_ave1s = 'H_avg = ' + str(format(h_avg, '.1f')) + 'mm'
                    cv2.putText(closing, str_H_ave1s, (int(photo_w * 0.05), int(photo_h * 0.95)), font,
                                int(0.8 / compression_H + 1), (0, 255, 0), int(2 / compression_H + 1))
                    # 图像，文字内容， 坐标 ，字体（横坐标，纵坐标），大小，颜色，字体厚度

        cv2.imshow('closing'+ str(id), imutils.resize(closing, width=640, height=480))

        key = cv2.waitKey(1)
        if int(key) == ord("s"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------------
if __name__ == '__main__':
    print("主进程开始启动！")
    for index in range(CAMERA_COUNT):
        print('摄像头的索引号是：', index)
        p = multiprocessing.Process(target=video_read, args=(index,))
        print("创建一个子进程")

        p.start()

    #p.join()

    print('主进程结束！')