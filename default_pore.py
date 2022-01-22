# -*- coding: utf-8 -*-
# encoding: utf-8
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def turn_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    colors = np.where(hist > 5000)
    img_number = 0
    for color in colors[0]:
        # print(color)
        split_image = img.copy()
        split_image[np.where(gray != color)] = 0
        cv2.imwrite(str(img_number) + ".jpg", split_image)
        img_number += 1
        return cv2.imshow("gray", split_image)


def circle(img):
    pic = cv2.medianBlur(img, 5)
    cimg = cv2.cvtColor(pic, cv2.COLOR_GRAY2BGR)
    circles = cv2.HoughCircles(pic, cv2.HOUGH_GRADIENT, 1, 100, param1=100, param2=30, minRadius=100, maxRadius=200)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(cimg, (i[0], i[1]), i[2] - 10, (0, 255, 0), 2)
    # crop boundary based on radius and center
    res = cimg[52:448, 64:460]
    return res


def binary(img):
    ret, th = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)
    return cv2.imshow("binary", th)


def contour(img):
    # turn binary bin
    ret, th = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)
    # get contours
    contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # draw rectangle using contours
    res = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    return res


def contourarea(img):
    area = []
    ret, th = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        if cv2.contourArea(cnt) >0 and cv2.contourArea(cnt)<500:
            mask = np.zeros_like(img)
            cv2.drawContours(mask, cnt, -1, 255, 2)
            cv2.imshow("test", mask)
            cv2.waitKey()
            area = cv2.contourArea(cnt)
            print(area)


def savimg(img):
    ret, th = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        if cv2.contourArea(c) < 1000:
            mask = np.zeros_like(img)
            # Create mask where white is what we want, black otherwise
            cv2.drawContours(mask, c, -1, 255, 2)  # Draw filled contour in mask
            ##             out = np.zeros_like(img) # Extract out the object and place into output image
            ##             out[mask == 255] = img[mask == 255]
            ##             (y, x) = np.where(mask == 255)
            ##             print([x,y])
            ##             print(len(x))
            ###             out = out[topy:bottomy+1, topx:bottomx+1]
            #            for i in mask:
            #                cv2.imwrite(os.path.join(path , 'crack'),i)
            cv2.imshow("test", mask)
            print("ok")
            cv2.waitKey()


# ITERATE OVER THE FOLDER
# C:\Users\Jerry\Downloads\JPG_test\JPG_test\20170106\9A\170106160410.17.01.06-1504504(17.01.06-1504504).613851dc\series_6
# C:\Users\Jerry\Downloads\JPG_test\JPG_test\20170109\9W\170109151816.17.01.09-1405904(17.01.09-1405904).a21500e0\series_4
# C:\Users\Jerry\Downloads\JPG_test\JPG_test\20170106\9W\series_5
# C:\Users\Jerry\Downloads\JPG_test\JPG_test\20170109\9A\170109151847.17.01.09-1405904(17.01.09-1405904).c71643ca\series_5
folder = (r"C:\Users\Jerry\Downloads\JPG_test\JPG_test\20170106\9A\170106160410.17.01.06-1504504(17.01.06-1504504).613851dc\series_6")
name = os.listdir(folder)
name.sort(key=lambda i: (len(i), i))
for i in range(len(name)):
    image = cv2.imread(folder + "\\" + name[i], 0)
    #binary(image)
    #cv2.waitKey()
    #print("ok")
    cv2.imshow("box",contour(image))
    cv2.waitKey()

    #contourarea(image)
    #savimg(image)
    #cv2.imshow("test", savimg(image))
    #cv2.waitKey()

#    cv2.imwrite(os.path.join(path , 'crack'+i),savimg(image))
#    cv2.waitKey(0)


# CREATE COLUMN
'''
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
u = np.linspace(0, 2 * np.pi, 50)
h = np.linspace(0, 196, 196)
x = np.outer(512 * np.sin(u), np.ones(len(h)))
y = np.outer(512 * np.cos(u), np.ones(len(h)))
z = np.outer(np.ones(len(u)), h)
# Plot the surface
ax.scatter(269.5, 215.5, 1, c="red")
ax.scatter(272.5, 215.5, 180, c="red")
ax.plot_surface(x, y, z, cmap=plt.get_cmap('bone'), alpha=0.5)
'''