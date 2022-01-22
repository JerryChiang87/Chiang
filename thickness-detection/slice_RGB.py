import cv2
import os
import numpy as np
p = r"C:\Users\Administrator\Desktop\Thickness data"
name = os.listdir(p)
name.sort(key=lambda i: (len(i), i))
ans = []
xx = 0
yy = 0
blue = []
green = []
red = []
mb = []
sb = []
mg = []
sg = []
mr = []
sr = []
def stat(sliced):
    for s in range(len(sliced)):
        #print(sliced[s].shape)
        for xx in range(100):
            #print(sliced[s][xx])
            for yy in range(100):
                #print(sliced[xx][yy])
                blue.append(sliced[s][xx][yy][0])
        mb.append(np.mean(blue))
    blue.clear()
    return mb
                  #print(blue)
#                green.append(sliced[xx][yy][1])
#                red.append(sliced[xx][yy][2])
                #print(blue)
#    return blue
    #print()

#0-33, 33-59,59-92
for a in range(59,92):
    img = cv2.imread(p + "\\" + name[a], 1)
    sliced = slice(img)
    blueres = stat(sliced)
    #stat(small[s])
    #stat(small[s])
#    stat(sliced)
#    print(len(sliced))
#print(len(stat(small[s])))
print(blueres)


