'''
简单的双进程展示双摄像头图像
按 s 结束一个进程运行，无法同时结束两个
'''
import numpy as np
import cv2
import multiprocessing
from multiprocessing import Queue
import time

def trackChaned(x):
  pass

CAMERA_COUNT = 2  # 摄像头个数
#url = "rtsp://admin:12345678@192.168.128.45"  # 外置摄像头地址及密码
# 不同进程不能共享内存，定义队列进行进程通信
q = Queue()


def video_read(id):
    camera_id = id

    exposure_value = 5

    # 使用笔记本自带的摄像头
    if camera_id == 0:
        cap = cv2.VideoCapture(0)

    # 使用外置的摄像头
    if camera_id == 1:
        cap = cv2.VideoCapture(1)
    cap_exposure = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
    print("摄像头曝光调节模式：")
    print(cap_exposure)
    # cap.set(21,0.25)
    # cap_exposure2 = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
    # print("摄像头曝光调节2模式：")
    # print(cap_exposure2)
    # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, -0.25)  #无法将摄像头设置为手动模式：0.25/-0.25都不行
    # cap_exposure2 = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
    # print("摄像头曝光调节3模式：")
    # print(cap_exposure2)
    cap.set(15, -(exposure_value))
    # 获取每一个视频的尺寸
    width = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(width, height)

    for i in range(49):
        if abs(cap.get(i) + 1) > 0.001:
             print("{}.parameter={}".format(i, cap.get(i)))

    cv2.namedWindow('Color Track Bar' + str(id))

    cv2.createTrackbar("Adj_exposure", 'Color Track Bar' + str(id), 4, 10, trackChaned)

    cv2.setTrackbarPos('Adj_exposure', 'Color Track Bar' + str(id), 5)

    while (cap.isOpened()):
        #ret, frame = cap.read()
        Adj_exposure = cv2.getTrackbarPos("Adj_exposure", 'Color Track Bar' + str(id))
        if (Adj_exposure != exposure_value):
            cap.set(15, -(Adj_exposure))
            cv2.setTrackbarPos('Adj_exposure', 'Color Track Bar' + str(id), Adj_exposure)
            exposure_value = Adj_exposure

        q.put(cap.read()[1])  # 刚刚放进去一张图片
        if q.qsize() > 1:
            q.get()
        else:
            time.sleep(0.001)  # 然后马上取出来
            #p.start()

        # for i in range(49):
        #     if abs(cap.get(i) + 1) > 0.001:
        #         print("{}.parameter={}".format(i, cap.get(i)))

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
        cv2.imshow('Color Track Bar' + str(id), frame)

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