import cv2
import os
import numpy as np
p = r"C:\Users\Administrator\Desktop\split"
splithsv = r"C:\Users\Administrator\Desktop\splithsv"
name = os.listdir(p)
name.sort(key=lambda i: (len(i), i))
count = 0
#0-33, 33-59,59-92
for a in range(len(name)):
    img = cv2.imread(p + "\\" + name[a], 1)
    #img = cv2.resize(img, (512,512))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imwrite(os.path.join(splithsv,"img" + str(count) + ".jpg"), hsv)
    count += 1
print("done")
    #cv2.imshow('Input', img)
    #cv2.imshow('Result', hsv)
    #cv2.waitKey()
    #print(img.shape)
    #for xi in range(512):
    #    for yi in range(512):
    #        ans.append(hsv[xi][yi][0])
    #m = np.mean(ans)
    #s = np.std(ans)
    #mean.append(m)
    #std.append(s)
    #ans = []
#print(mean)
#print(std)