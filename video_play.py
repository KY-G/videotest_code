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
。。。。。。。。。。。。。。。。。。。2020.02.01。。。。。。。。。。。。。。。。。。。。
计划

汉字显示
'''
import cv2
import numpy as np

def trackChaned(x):
  pass

def Contrast_and_Brightness(img):

    con = cv2.getTrackbarPos("Adj_contrast", "Color_con_bri Track Bar")*0.1
    bri = cv2.getTrackbarPos("Adj_brightness", "Color_con_bri Track Bar")*0.1

    blank = np.zeros(img.shape, img.dtype)
    dst = cv2.addWeighted(img, con, blank, 1-con, bri)
#    cv2.imshow("video", frame)
    return dst

cap = cv2.VideoCapture(0)      # 参数为设备索引号，0代表内置摄像头
#cap_2 = cv2.VideoCapture(1)
j = 0
w0 = 0
h0 = 0
w_ave = 0
h_ave = 0
compression_H = 0.32
compression_W = 0.32

cap.set(3, 2000)  # 宽
cap.set(4, 1500)  # 高
#曝光度设置
cap.set(15, -4)
print("exposure={}".format(cap.get(15)))

cv2.namedWindow('Color_con_bri Track Bar')

cv2.createTrackbar("Adj_gray", "Color_con_bri Track Bar", 0, 255, trackChaned)
cv2.createTrackbar("Adj_contrast", "Color_con_bri Track Bar", 0, 20, trackChaned)
cv2.createTrackbar("Adj_brightness", "Color_con_bri Track Bar", 0, 1000, trackChaned)

# 设置默认值
cv2.setTrackbarPos('Adj_gray', 'Color_con_bri Track Bar', 78)
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

    height_clo, width_clo = frame.shape[:2]
    frame_1 = cv2.resize(frame, (int(width_clo * compression_W), int(height_clo * compression_H)))
    cv2.imshow('video', frame_1)

    gray = cv2.cvtColor(Contrast_and_Brightness(frame), cv2.COLOR_BGR2GRAY)

    Adj = cv2.getTrackbarPos("Adj_gray", "Color_con_bri Track Bar")

    ret, binary = cv2.threshold(gray, Adj, 255, cv2.THRESH_BINARY)

    closing = binary
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    img3, contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(0, len(contours)):
        cnt = contours[i]
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.rectangle(closing, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.drawContours(gray, contours, -1, (0, 255, 0), 3)
        cv2.drawContours(closing, contours, -1, (0, 0, 255), 3)
        size = closing.shape
        ww = size[1]  # 宽度
        hh = size[0]  # 高度

        xw = w * 0.0215
        yh = h * 0.0215

        if xw >= 1 and yh <= 8 and xw < 2*yh < 3*xw:
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

                # # 设置需要显示的字体
                # fontpath = 'simhei.ttf' #< br >字体设定？？ # 32为字体大小
                # font = ImageFont.truetype(fontpath, 32, encoding="utf-8")
                # img_pil = Image.fromarray(closing)
                # draw = ImageDraw.Draw(img_pil)
                # # 绘制文字信息<br># (100,300/350)为字体的位置，(255,255,255)为白色，(0,0,0)为黑色
                # draw.text((100, 300), "你好", font=font, fill=(0, 255, 0))
                # closing = np.array(img_pil)

                font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体,读取到多个图形时会导致文字重叠
                ss = 'W = ' + str(format(xw, '.4f')) + 'mm'
                W_ave1s = 'Wave_1s = ' + str(format(w_ave, '.4f')) + 'mm'
                cv2.putText(closing, ss, (int(ww * 0.2), 150), font, 2.4, (0, 255, 0), 8)
                cv2.putText(closing, W_ave1s, (10, 1300), font,2.4, (0, 255, 0), 8)
                                            #文字在屏幕中的位置，
                # 图像，文字内容， 坐标（横坐标，纵坐标） ，字体，大小，颜色，字体厚度

                ss = 'H = ' + str(format(yh, '.4f')) + 'mm'
                H_ave1s = 'Have_1s = ' + str(format(h_ave, '.4f')) + 'mm'
                cv2.putText(closing, ss, (int(ww * 0.6), 150), font, 2.4, (0, 255, 0), 8)
                cv2.putText(closing, H_ave1s, (10, 1400), font, 2.4, (0, 255, 0), 8)
                # 图像，文字内容， 坐标 ，字体（横坐标，纵坐标），大小，颜色，字体厚度


    height_clo, width_clo = closing.shape[:2]
    closing_1 = cv2.resize(closing, (int(width_clo *compression_W), int(height_clo *compression_H)))
    cv2.imshow('closing_1', closing_1)

    if cv2.waitKey(1) == ord("s"):  # 按's' 退出，也可以设置其他键，用ord()转换为ASCIIwaitKey()返回ASCII码
        break

cap.release()
cv2.destroyAllWindows()
