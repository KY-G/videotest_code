'''

瞳孔直径检测演示程序，单目
每隔1s发送一个平均瞳孔值
可拖动调节二值化阈值
加入亮度、对比度拖动调节
按 s 结束运行
循坏外进行分辨率调节，实现完全显示摄像头图像，处理时使用完全图像，显示时采用压缩显示（原图和clossing)
读出的瞳孔直径数值优化，去掉小数点(现保留一位小数点)
加入局部扫描（现扫描范围为大于1/3，小,于2/3）
瞳孔平均值采用滑动平均显示，20个值求平均。当瞳孔消失在检测范围内显示的数据也会同步消失。
瞳孔边缘查找采用 _LIST 模式（findcon()不分级，这样效果较好）。边缘查找效果好，文字重叠问题可以得到优化，但恶劣情况下仍会发生。

解决读出文字叠加问题。
    循环显示读出的边缘列表可能是导致文字叠加的原因
    给边缘查找的数组长度加限制，len(counter)限制在 2 时是最优的结果（可以扩展到4，但不建议更大。）
    但会导致屏幕出现瞳孔但不会显示直径。
平均值采用滑动平均值，实时显示
    平均值实现始终在屏幕上显示问题
已加入曝光设置（-1-- -10）
瞳孔直径显示已校准，3-5mm的瞳孔显示较准
将滑动调节和视频显示框做到一起显示
。。。。。。。。。。。。。。。。。。。2020.02.01。。。。。。。。。。。。。。。。。。。。
计划
汉字显示


'''
import cv2
import numpy as np
import imutils
import tkinter as tk
from PIL import ImageFont, ImageDraw, Image

def trackChaned(x):
  pass

def Contrast_and_Brightness(img):

    con = cv2.getTrackbarPos("Adj_contrast", "Color_con_bri Track Bar")*0.1
    bri = cv2.getTrackbarPos("Adj_brightness", "Color_con_bri Track Bar")*0.1

    blank = np.zeros(img.shape, img.dtype)
    dst = cv2.addWeighted(img, con, blank, 1-con, bri)
#    cv2.imshow("video", frame)
    return dst

def paint_chinese_opencv(img,chinese,position,fontsize,color):#opencv输出中文
    font = ImageFont.truetype("font/simsun.ttc", fontsize)
    img_PIL = Image.fromarray(img)  # 图像从OpenCV格式转换成PIL格式
    # color = (255,0,0) # 字体颜色
    # position = (100,100)# 文字输出位置
    draw = ImageDraw.Draw(img_PIL)
    draw.text(position, chinese, font=font, fill=color)  # PIL图片上打印汉字 # 参数1：打印坐标，参数2：文本，参数3：字体颜色，参数4：字体
    img = np.array(img_PIL)
    return img

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

cap = cv2.VideoCapture(0)      # 参数为设备索引号，0代表内置摄像头
#cap_2 = cv2.VideoCapture(1)
j = 0
list_w = []
list_h = []
list_long = 20 #求平均瞳孔直径时的采集值个数
# w0 = 0
# h0 = 0
# w_ave = 0
# h_ave = 0
# compression_H = 0.25
# compression_W = 0.25
compression_H = 0.32
compression_W = 0.32

k = 0
Adj_gray = 10
adj_over = 0
list_contours = []
list_contours_long = 10

# cap.set(3, 2580)  # 宽
# cap.set(4, 1940)  # 高
cap.set(3, 2000)  # 宽
cap.set(4, 1500)  # 高
#曝光度设置
cap.set(15, -4)
print("exposure={}".format(cap.get(15)))

cv2.namedWindow('Color_con_bri Track Bar')
# hh = 'Adj_gray'
# #hl = 'Min'
# wnd = 'Colorbars'
cv2.createTrackbar("Adj_gray", "Color_con_bri Track Bar", 0, 255, trackChaned)
cv2.createTrackbar("Adj_contrast", "Color_con_bri Track Bar", 0, 20, trackChaned)
cv2.createTrackbar("Adj_brightness", "Color_con_bri Track Bar", 0, 1000, trackChaned)

# 设置默认值
cv2.setTrackbarPos('Adj_gray', 'Color_con_bri Track Bar', Adj_gray)
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

while(True):
    ret, frame = cap.read()     # 返回一个布尔值(ret)
    frame = cv2.flip(frame, 1)  # 若无本语句，摄像头非镜像
    c = cv2.waitKey(10)

#亮度bla、对比度con调节
    #cv2.imshow('video_1', frame)
    # height_clo, width_clo = frame.shape[:2]
    # frame_1 = cv2.resize(frame, (int(width_clo * compression_W), int(height_clo * compression_H)))
    # cv2.imshow('video', frame_1)
    cv2.imshow("Color_con_bri Track Bar", imutils.resize(frame, width=640, height=480))  # 重新定义分辨率
    gray = cv2.cvtColor(Contrast_and_Brightness(frame), cv2.COLOR_BGR2GRAY)

    Adj = cv2.getTrackbarPos("Adj_gray", "Color_con_bri Track Bar")

    ret, binary = cv2.threshold(gray, Adj, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    img3, contours, hierarchy = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                                                        #第二个参数由cv2.RETR_TREE改为RETE_LIST,不分等级
    size = closing.shape
    photo_w = size[1]  # 宽度
    photo_h = size[0]  # 高度

    if k < 5:
        list_contours.append(len(contours))
        k = +1
    else:
        k = 0
    # 删除数组首位元素
    if len(list_contours) == (list_contours_long + 1) :
        del list_contours[0]  # 删除数组首位

    # 将图片中像素以数列的形式全部打印输出
    pixel_data = np.array(closing)  #
    # print(pixel_data)
    a = np.sum(pixel_data)  # 数组所有元素求和。 nan与数字运算还是nan
    pixel_avg = a / (closing.size)
    # print("像素和：%s" % a)
    # print("像素个数：%s" % closing.size)  # 像素个数
    #print("平均值：%s" % (a / (closing.size)))
    if(pixel_avg <= 220)&(adj_over == 0):
        adj_over = 0
        Adj_gray = Adj_gray - 100
        cv2.setTrackbarPos('Adj_gray', 'Color_con_bri Track Bar', Adj_gray)
        #pixel_avg = (np.sum(np.array(closing))) / (closing.size)
    elif (pixel_avg >= 253)&(adj_over == 0):
        adj_over = 0
        Adj_gray = Adj_gray + 30
        cv2.setTrackbarPos('Adj_gray', 'Color_con_bri Track Bar', Adj_gray)
        #pixel_avg = (np.sum(np.array(closing))) / (closing.size)
    else :
        if (len(contours) > 4) & (adj_over == 0):
            adj_over = 0
            Adj_gray = Adj_gray - 10
            cv2.setTrackbarPos('Adj_gray', 'Color_con_bri Track Bar', Adj_gray)
            #pixel_avg = (np.sum(np.array(closing))) / (closing.size)
        elif(average(list_contours)<4) :
            adj_over = 1
    #长度最好的状态为len(contours) = 2时，可以放宽到4？？
    if adj_over == 1 :
        for i in range(0, len(contours)):
            cnt = contours[i]
            x, y, w, h = cv2.boundingRect(cnt)
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), int(1/compression_H))
            cv2.rectangle(closing, (x, y), (x + w, y + h), (0, 255, 0), int(1 / compression_H))
            # cv2.drawContours(gray, contours, -1, (0, 255, 0), int(1/compression_H))
            pupil_w = w * 0.0215
            pupil_h = h * 0.0215

            if (8 >= pupil_w >= 1 and 1 <= pupil_h <= 8) and (pupil_w < 2 * pupil_h < 3 * pupil_w):
                if (photo_w * 0.7 > x > photo_w * 0.3) and (photo_h * 0.6 > y > photo_h * 0.3):
                    cv2.drawContours(closing, contours, -1, (0, 0, 255), int(1 / compression_H))
                    # 写入数组数据
                    if j < 20:
                        list_w.append(pupil_w)  #从数组末端加入
                        list_h.append(pupil_h)
                        j = +1
                    else:
                        j = 0
                    # 删除数组首位元素
                    if len(list_w) == list_long+1 | len(list_h) == list_long+1:
                        del list_w[0]   #删除数组首位
                        del list_h[0]

                    print('w = %.4f' % pupil_w, 'mm')  # 整数‘%d’
                    print('h = %.4f' % pupil_h, 'mm')
                    print(type(contours))
                    print(len(contours))

                   # print("Updated list_w : ", list_w) #显示数组所有的元素
                    w_avg = average(list_w)
                    h_avg = average(list_h)

                    # # 设置需要显示的字体
                    # fontpath = 'simhei.ttf' #< br >字体设定？？ # 32为字体大小
                    # font = ImageFont.truetype(fontpath, 32, encoding="utf-8")
                    # img_pil = Image.fromarray(closing)
                    # draw = ImageDraw.Draw(img_pil)
                    # # 绘制文字信息<br># (100,300/350)为字体的位置，(255,255,255)为白色，(0,0,0)为黑色
                    # draw.text((100, 300), "你好", font=font, fill=(0, 255, 0))
                    # closing = np.array(img_pil)

                    font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体,读取到多个图形时会导致文字重叠
                    str_w = 'Width_pupil = ' + str(format(pupil_w, '.1f')) + 'mm'
                    str_W_ave1s = 'W_avg = ' + str(format(w_avg, '.1f')) + 'mm'
                    cv2.putText(closing, str_w, (int(photo_w * 0.01), int(photo_h * 0.1)), font,
                                int(0.6 / compression_H + 1), (0, 255, 0), int(2 / compression_H + 1))
                    cv2.putText(closing, str_W_ave1s, (int(photo_w * 0.05), int(photo_h * 0.85)), font,
                                int(0.8 / compression_H + 1), (0, 255, 0), int(2 / compression_H + 1))
                    # 文字在屏幕中的位置，
                    # 图像，文字内容， 坐标（横坐标，纵坐标） ，字体，大小，颜色，字体厚度

                    str_h = 'High_pupil = ' + str(format(pupil_h, '.1f')) + 'mm'
                    str_H_ave1s = 'H_avg = ' + str(format(h_avg, '.1f')) + 'mm'
                    cv2.putText(closing, str_h, (int(photo_w * 0.41), int(photo_h * 0.1)), font,
                                int(0.6 / compression_H + 1), (0, 255, 0), int(2 / compression_H + 1))
                    cv2.putText(closing, str_H_ave1s, (int(photo_w * 0.05), int(photo_h * 0.95)), font,
                                int(0.8 / compression_H + 1), (0, 255, 0), int(2 / compression_H + 1))
                    # 图像，文字内容， 坐标 ，字体（横坐标，纵坐标），大小，颜色，字体厚度

            else:
                w_avg = average(list_w)
                h_avg = average(list_h)
                font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体,读取到多个图形时会导致文字重叠
                #str_w = 'Width_pupil error !!! '
                str_W_ave1s = 'W_avg = ' + str(format(w_avg, '.1f')) + 'mm'
                #cv2.putText(closing, str_w, (int(photo_w * 0.01), int(photo_h * 0.1)), font,
                            #int(0.6 / compression_H + 1), (0, 255, 0), int(2 / compression_H + 1))
                cv2.putText(closing, str_W_ave1s, (int(photo_w * 0.05), int(photo_h * 0.85)), font,
                            int(0.8 / compression_H + 1), (0, 255, 0), int(2 / compression_H + 1))
                # 文字在屏幕中的位置，
                # 图像，文字内容， 坐标（--横坐标，纵坐标） ，字体，大小，颜色，字体厚度

                #str_h = 'High_pupil error !!! '
                str_H_ave1s = 'H_avg = ' + str(format(h_avg, '.1f')) + 'mm'
                #cv2.putText(closing, str_w, (int(photo_w * 0.41), int(photo_h * 0.1)), font,
                            #int(0.6 / compression_H + 1), (0, 255, 0), int(2 / compression_H + 1))
                cv2.putText(closing, str_H_ave1s, (int(photo_w * 0.05), int(photo_h * 0.95)), font,
                            int(0.8 / compression_H + 1), (0, 255, 0), int(2 / compression_H + 1))
                # 图像，文字内容， 坐标 ，字体（横坐标，纵坐标），大小，颜色，字体厚度
    else:
        w_avg = average(list_w)
        h_avg = average(list_h)
        font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体,读取到多个图形时会导致文字重叠
        #str_w = 'Width_pupil error !!! '
        str_W_ave1s = 'W_avg = ' + str(format(w_avg, '.1f')) + 'mm'
        #cv2.putText(closing, str_w, (int(photo_w * 0.01), int(photo_h * 0.1)), font,
                    #int(0.6 / compression_H + 1), (0, 255, 0), int(2 / compression_H + 1))
        cv2.putText(closing, str_W_ave1s, (int(photo_w * 0.05), int(photo_h * 0.85)), font,
                    int(0.8 / compression_H + 1), (0, 255, 0), int(2 / compression_H + 1))
        # 文字在屏幕中的位置，
        # 图像，文字内容， 坐标（横坐标，纵坐标） ，字体，大小，颜色，字体厚度

        #str_h = 'High_pupil error !!! '
        str_H_ave1s = 'H_avg = ' + str(format(h_avg, '.1f')) + 'mm'
        #cv2.putText(closing, str_w, (int(photo_w * 0.41), int(photo_h * 0.1)), font,
                    #int(0.6 / compression_H + 1), (0, 255, 0), int(2 / compression_H + 1))
        cv2.putText(closing, str_H_ave1s, (int(photo_w * 0.05), int(photo_h * 0.95)), font,
                    int(0.8 / compression_H + 1), (0, 255, 0), int(2 / compression_H + 1))
        # 图像，文字内容， 坐标 ，字体（横坐标，纵坐标），大小，颜色，字体厚度

        str_pixel_avg = str(format(pixel_avg, '.4f'))
        cv2.putText(closing, str_pixel_avg, (int(photo_w * 0.6), int(photo_h * 0.3)), font,
                    int(0.8 / compression_H + 1), (0, 255, 0), int(2 / compression_H + 1))

    # height_clo, width_clo = closing.shape[:2]
    # closing_1 = cv2.resize(closing, (int(width_clo *compression_W), int(height_clo *compression_H)))
    # cv2.imshow('closing_1', closing_1)
    cv2.imshow('closing_1', imutils.resize(closing, width=640, height=480))

    if cv2.waitKey(1) == ord("s"):  # 按's' 退出，也可以设置其他键，用ord()转换为ASCIIwaitKey()返回ASCII码
        break

'''
加上像素重新设置处理和压缩的情况下一个周期900ms
分辨率设置最耗时间660ms
第二位的是摄像头图片读取，图片处理到时不太耗时间
紧接着就是压缩图片输出  20ms左右

不加像素重新设置处理的情况下一个循环大概340ms（？？）
其他图像操作基本上10ms以内都可以处理完。加上延时50ms大概60ms一个周期

import cv2
from PIL import ImageFont, ImageDraw, Image
import numpy as np
<br># 编辑图片路径
bk_img = cv2.imread("background.jpg")
#设置需要显示的字体
fontpath = "font/simsun.ttc"<br># 32为字体大小
font = ImageFont.truetype(fontpath, 32)
img_pil = Image.fromarray(bk_img)
draw = ImageDraw.Draw(img_pil)
#绘制文字信息<br># (100,300/350)为字体的位置，(255,255,255)为白色，(0,0,0)为黑色
draw.text((100, 300),  "Hello World", font = font, fill = (255, 255, 255))
draw.text((100, 350),  "你好", font = font, fill = (255, 255, 255))
bk_img = np.array(img_pil)
 
cv2.imshow("add_text",bk_img)
cv2.waitKey()<br># 保存图片路径
cv2.imwrite("add_text.jpg",bk_img)　　
'''

cap.release()
cv2.destroyAllWindows()
