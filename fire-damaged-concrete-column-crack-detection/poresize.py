import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
def poresize(img):
    ret, th = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area.append(cv2.contourArea(cnt))
    return area

folder = (r'C:\Users\Jerry\Downloads\JPG_test\JPG_test\20170106\3A\170106160311.17.01.06-1504504(17.01.06-1504504).913c8bc9\series_4')
name = os.listdir(folder)
name.sort(key=lambda i: (len(i), i))
area = []
image = []
for i in range(len(name)):
    image.append(cv2.imread(folder + "\\" + name[i], 0))
for j in range(len(image)):
    ret, th = cv2.threshold(image[j], 240, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(th, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        if cv2.contourArea(cnt) > 0 and cv2.contourArea(cnt) < 5000:
            #print(cv2.contourArea(cnt))
            #cv2.drawContours(image[j], cnt, -1, 0, 2)
            #cv2.imshow("test", image[j])
            #cv2.waitKey()
            area.append(cv2.contourArea(cnt))
for i in range(len(area)):
    print(area[i])
