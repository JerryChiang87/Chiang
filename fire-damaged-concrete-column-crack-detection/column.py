# -*- coding: utf-8 -*-
"""
Created on Mon May  3 09:39:05 2021

@author: Jerry
"""

import cv2
import numpy as np
import os
from os import listdir
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# get the filenames in directory
basepath = r"C:\Users\Jerry\Downloads\JPG_test\JPG_test\20170106\9A\170106160410.17.01.06-1504504(17.01.06-1504504).613851dc\series_6"
files = listdir(basepath)
files.sort(key=lambda i: (len(i), i))
path = r"C:\Users\Jerry\Downloads\crack"

fig = plt.figure()
fig.set_size_inches(20, 20)
ax = fig.add_subplot(111, projection='3d')
u = np.linspace(0, 2 * np.pi, 100)
h = np.linspace(0, 298, 149)
a = np.outer(198 * np.sin(u), np.ones(len(h))) + 262
b = np.outer(198 * np.cos(u), np.ones(len(h))) + 250
c = np.outer(np.ones(len(u)), h)
ax.set_xlim(0, 512)
ax.set_ylim(0, 512)
ax.set_xlabel('Pixels')
ax.set_ylabel('Pixels')
ax.set_zlabel('Pixels')
ax.plot_surface(a, b, c, cmap=plt.get_cmap('bone'), alpha=0.2)
z = 0
cracknum = 0
for i in files:
    image = cv2.imread(basepath + "/" + i, 0)
    # threshold image

    ret, mask = cv2.threshold(image, 230, 255, cv2.THRESH_BINARY)
    # find contours
    contours, hier = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    #print(len(contours))

    for j in range(len(contours)):
        [x1, y1, w1, h1] = cv2.boundingRect(contours[j])
        ratio = w1 / h1

        if cv2.contourArea(contours[j]) < 800 and cv2.contourArea(contours[j]) > 100 and (ratio > 2 or ratio < 0.5):
            cnt = contours[j]
            cracknum += 1
            for k in cnt:
                for p in k:
                    x = p[0]
                    y = p[1]
                    # ax.scatter(x,y,z,c="red")
                    # ax.contourf()
                    ax.scatter(x, y, z, c="red", marker="|", linewidths=1)

        #if cv2.contourArea(contours[j])>50 and cv2.contourArea(contours[j])<100 and (ratio < 1.5 or ratio >0.5):
        #    cnt = contours[j]
        #    for k in cnt:
        #        for p in k:
        #            x = p[0]
        #            y = p[1]
        #            ax.scatter(x,y,z,c="gray", marker = 1 , linewidths = 0.3)
        #    plt.pause(0.0000001)

    z += 2
plt.show()
print(cracknum)
#        mask = np.zeros_like(cnt)
#        cv2.drawContours(mask, cnt, -1, 255, 2)
#        cv2.imshow("aa",mask)
#        cv2.waitKey(0)
# get contour
#        cnt = contours[j]
#        for k in cnt:
#        # get the dimensions of the boundingRect
#        # save image of contour with indexed name
#            cv2.imwrite(path+"\contour_"+str(i)+".jpg", contours[j][k])
