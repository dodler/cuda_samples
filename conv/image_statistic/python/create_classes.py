from os.path import isfile, join

import numpy as np
import cv2
from matplotlib import pyplot as plt
from os import listdir

# print len("_10_10.output.txt") 17
# print len("1-a.png_10_10_.txt") 18

all_signs_path = "/home/lyan/Documents/image_statistic/100/"

suffix_len = 17

all_signs = [f for f in listdir(all_signs_path) if isfile(join(all_signs_path, f))]

classes = {}
for s in all_signs:
    cls_name = s[0:len(s) - suffix_len] #take a closer look at structure of file name, that contains sign valus
    # for current pic
    if classes.has_key(cls_name):
        classes[cls_name] += 1
    else:
        classes[cls_name] = 1

print len(classes)

signs_num = 13

classes_signs_centers = {}

for i in range(0, len(all_signs)):  # i is index for current picture signs values
    with open(all_signs_path + all_signs[i]) as f:
        cls_name = all_signs[i][0:len(all_signs[i]) - suffix_len]
        lines = f.readlines()  # current image signs values
        for j in range(0, signs_num):  # j is sign index (from 0 to 13 or 14)
            value = np.float32(lines[j])  # j th sign value for i th image

            if np.isnan(value) or np.isinf(value):
                value = 0

            if classes_signs_centers.has_key(cls_name):
                classes_signs_centers[cls_name][j] += value
            else:
                classes_signs_centers[cls_name] = [0]*signs_num
                classes_signs_centers[cls_name][j] = value;

for j in range(0, signs_num):
    for cls_name in classes:
        classes_signs_centers[cls_name][j] /= classes[cls_name]

print classes_signs_centers

path = "/home/lyan/Documents/vkr/output/20_20/output/"
common_signs = [f for f in listdir(path) if isfile(join(path, f))]

lines = 0

with open(path + common_signs[0]) as f:
    lines = len(f.readlines())

res = list()

clusterNum = 27
maxIter = 20
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

min_comp = {}

for i in range(0, len(common_signs)):
    with open(path + common_signs[i]) as f:
        output = np.float32(f.readlines())
        output[np.isnan(output)] = 0
        output[np.isinf(output)] = 0
        retval, bestLabels, centers = cv2.kmeans(output, clusterNum, criteria, maxIter, flags=cv2.KMEANS_PP_CENTERS)
        # class_name = str(j)
        # print common_signs[i]

        centers_avg = sum(centers)
        centers_avg /= len(centers)
        centers_value_max = max(centers_avg)

        er = 0
        for s in output:
            er += (centers_avg-s)/centers_value_max

        print "er:",er

        # if min_comp.has_key(class_name):
        #     if (retval < min_comp[class_name]):
        #         min_comp[class_name] = retval
        # else:
        #     min_comp[class_name] = retval
