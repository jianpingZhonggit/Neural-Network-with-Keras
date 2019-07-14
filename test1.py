# -*- coding: utf-8 -*-
import cv2
import os


def find_circle(path):
    # 载入并显示图片
    img = cv2.imread(path)
# cv2.namedWindow('1', 0)
# cv2.imshow('1', img)
# cv2.waitKey(0)
    img = img[int(img.shape[0]/2):img.shape[0]]
    print(img.shape)
    # 降噪（模糊处理用来减少瑕疵点）
    result = cv2.blur(img, (5, 5))
# cv2.namedWindow('2', 0)
# cv2.imshow('2', result)
# cv2.waitKey(0)
# 灰度化,就是去色（类似老式照片）
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
# cv2.namedWindow('3', 0)
# cv2.imshow('3', gray)
# cv2.waitKey(0)

    # param1的具体实现，用于边缘检测
    canny = cv2.Canny(img, 40, 80)
# cv2.namedWindow('4', 0)
# cv2.imshow('4', canny)
# cv2.waitKey(0)

    # 霍夫变换圆检测
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 50, param1=80, param2=30, minRadius=0, maxRadius=120)
    # 输出返回值，方便查看类型
    print("+++", circles, "++++")

    # 输出检测到圆的个数
    print(len(circles[0]))

    print('-------------我是条分割线-----------------')
    # 根据检测到圆的信息，画出每一个圆
    for circle in circles[0]:
        # 圆的基本信息
        print(circle[2])
        # 坐标行列(就是圆心)
        x = int(circle[0])
        y = int(circle[1])
        # 半径
        r = int(circle[2])
        # 在原图用指定颜色圈出圆，参数设定为int所以圈画存在误差
        img = cv2.circle(img, (x, y), r, (0, 0, 255), 1, 8, 0)
    # 显示新图像
    cv2.namedWindow('5', 0)
    cv2.imshow('5', img)
    cv2.waitKey(0)
    # 按任意键退出


if __name__ == "__main__":
    img_list = os.listdir('./temp/')
    for img_path in img_list:
        find_circle('./temp/'+img_path)
    cv2.destroyAllWindows()
