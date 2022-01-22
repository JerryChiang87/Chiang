import cv2
import os
import numpy as np
p = r"data-3clearcat/train/uncoated"
split = r"data-3clearcat/train/uncoated"
name = os.listdir(p)
name.sort(key=lambda i: (len(i), i))
#print(len(name))
#img = []
#small = []
count = 0
for num in range(len(name)):
    img = cv2.imread(p + "\\" + name[num], 1)
    #img = cv2.resize(img, [2240,2240])
    height = img.shape[0]
    width = img.shape[1]
#print(height, width)
    height_cutoff = height // 10
    width_cutoff = width // 10
#print(width_cutoff)
    up = 0
    for i in range(10):
        down = up + height_cutoff
        left = 0
        for j in range(10):
            right = left + width_cutoff
            #small.append(img[up:down,left:right])
            cv2.imwrite(os.path.join(split, "img" + str(count) + ".jpg"), img[up:down, left:right])
            left += width_cutoff
            #cv2.imshow(str(count), small[j])
            #cv2.waitKey()
            #cv2.destroyAllWindows()
            count += 1
            #print(count)
        up += height_cutoff
