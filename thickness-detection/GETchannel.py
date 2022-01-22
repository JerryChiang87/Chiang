import cv2
import os
import numpy as np
import warnings
#warnings.filterwarnings("ignore", category=RuntimeWarning)
p = r"C:\Users\Administrator\Desktop\splithsv"
name = os.listdir(p)
name.sort(key=lambda i: (len(i), i))
#print(name)
b = []
g = []
r = []
mb = []
mg = []
mr = []
sb = []
sg = []
sr = []
h = []
mh = []
sh = []
for a in range(70000,82800):
    img = cv2.imread(p + "\\" + name[a], 1)
    #cv2.imshow('test', img)
    #cv2.waitKey()
    #0-33, 33-59,59-92
    #height, width, depth = img.shape
    for xi in range(100):
        for yi in range(100):
            h.append(img[xi,yi][0])
            b.append(img[xi,yi][0])
            g.append(img[xi,yi][1])
            r.append(img[xi,yi][2])
    mh.append(np.mean(h))
    sh.append(np.std(h))
    h.clear()
    mb.append(np.mean(b))
    sb.append(np.std(b))
    mg.append(np.mean(g))
    sg.append(np.std(g))
    mr.append(np.mean(r))
    sr.append(np.std(r))
    b.clear()
    g.clear()
    r.clear()
    #print(count)
print(mh)
print(sh)
print(mb)
print(sb)
print(mg)
print(sg)
print(mr)
print(sr)





