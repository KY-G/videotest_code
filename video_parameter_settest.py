import cv2

# 选择摄像头号，一般从 0 开始
cap = cv2.VideoCapture(0)
cap.set(15, -7)
# 先设置参数，然后读取参数
for i in range(47):
    print("No.={} parameter={}".format(i, cap.get(i)))

while True:
    ret, img = cap.read()
    cv2.imshow("input", img)
    # 按 ESC 键退出
    key = cv2.waitKey(10)
    if cv2.waitKey(1) == ord("s"):  # 按's' 退出，也可以设置其他键，用ord()转换为ASCIIwaitKey()返回ASCII码
        break

cv2.destroyAllWindows()
cv2.VideoCapture(0).release()