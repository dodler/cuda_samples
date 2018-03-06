from os.path import isfile, join

import numpy as np
import cv2
from matplotlib import pyplot as plt
from os import listdir

import collections

class_value = collections.namedtuple('ClassValue',['value','name'])

test = class_value("1", "2")
print test

path = "/home/lyan/Documents/vkr/output/50_50/output/"
all_signs_path = "/home/lyan/Documents/vkr/output/50_50/"

#0.0384615384615 10 10
#0.0380952380952 20 20
#0.0380952380952 30 30
#0.0380952380952 40 40
#0.0384615384615 50 50

suffix_len = 17
sign_index = 1

all_signs_cls_value = [];
all_signs = [f for f in listdir(all_signs_path) if isfile(join(all_signs_path, f))]
for f in all_signs:
    with open(all_signs_path + f) as fl:
        lines = fl.readlines()
        all_signs_cls_value.append(class_value(lines[sign_index],f[0:len(f)-suffix_len]))

print len(all_signs_cls_value)
print all_signs_cls_value

files = [f for f in listdir(path) if isfile(join(path, f))]

lines = 0

with open(path + files[0]) as f:
    lines = len(f.readlines())

res = list()

clusterNum = 27
maxIter = 20
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

classes = {}

f_num = 3

with open(path + files[f_num]) as f:
    output = np.float32(f.readlines())
    output[np.isnan(output)] = 0
    output[np.isinf(output)] = 0
    comp, bestLabels, centers = cv2.kmeans(output, clusterNum, criteria, maxIter, flags=cv2.KMEANS_PP_CENTERS)

    classes = {}
    classes_cnt = 0

    for cls_value in all_signs_cls_value:
        for c in centers:
            # print cls_value[0]
            o = np.float32(cls_value[0])
            if (o <= c + comp) or (o >= c - comp):
                if classes.has_key(str(c)):
                    classes[str(c)].append(cls_value)
                else:
                    classes[str(c)] = [] # here lay classes distributed by cluster name

for k in classes.keys():
    cls_values = classes[k]
    class_cnt = {}

    for cls_value in cls_values:
        cls_name = cls_value[1]
        if class_cnt.has_key(cls_name):
            class_cnt[cls_name] += 1
        else:
            class_cnt[cls_name] = 1

    max = 0
    name = ''

    for cls in class_cnt.keys():
        if np.int(class_cnt[cls]) > max:
            max = np.int(class_cnt[cls])
            name = cls

    print name
    classes[name] = classes.pop(k)

for cls in classes.keys():
    cls_names = classes[cls]
    classes_names_len = len(cls_names)
    cnt = 0.0
    for cn in cls_names:
        if cn[1] == cls:
            cnt += 1
    print "1"
    print cnt / classes_names_len

print classes


