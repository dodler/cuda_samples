import traceback
from os.path import isfile, join

import numpy as np
from pandas import unique
from string import replace

import cv2
from matplotlib import pyplot as plt
from os import listdir

import collections

class_value = collections.namedtuple('ClassValue', ['value', 'name', 'label'])

def get_file_name_by_sign_index(index, dm1, dm2):
    return str(index) + "_" + str(dm1) + "_" + str(dm2) + ".output.txt"

suffix_len = 17
sign_index = 2
dm1 = 50
dm2 = 50

def calc_error():

    def prefix(dm1, dm2):
        return str(dm1) + "_" + str(dm2)

    path = "/home/lyan/Documents/vkr/noisy/" + prefix(dm1, dm2) + "/output/"
    all_signs_path = "/home/lyan/Documents/vkr/noisy/" + prefix(dm1, dm2) + "/"

    all_signs_cls_value = [];
    all_signs = [f for f in listdir(all_signs_path) if isfile(join(all_signs_path, f))]
    for f in all_signs:
        with open(all_signs_path + f) as fl:
            lines = fl.readlines()
            lines = [replace(l, '\n', '') for l in lines]
            all_signs_cls_value.append(class_value(lines[sign_index-1], f[0:len(f) - suffix_len], str(0)))

    clusterNum = 27
    maxIter = 20
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    with open(path + get_file_name_by_sign_index(sign_index, dm1, dm2)) as f:
        lines = f.readlines()
        lines = [replace(l, '\n', '') for l in lines]

        output = np.float32(lines)

        output[np.isnan(output)] = 0
        output[np.isinf(output)] = 0
        comp, labels, centers = cv2.kmeans(output, clusterNum, criteria, maxIter, flags=cv2.KMEANS_PP_CENTERS)

        cluster_values = {}

        for l in unique(labels):
            cluster_values[str(l)] = []
            for i in range(0, len(labels)):
                cluster_values[str(l)].append(class_value(lines[i], '', l))

        # print "all_signs_cls_value=", all_signs_cls_value

        # t = [getattr(sgn,'value')=='5.9721o9e+08' for sgn in all_signs_cls_value]
        # print t.index(True)

        clusters_output = []
        for clv in cluster_values.keys():
            cl_values = cluster_values[clv]
            for cl_value in cl_values:

                try:
                    print cl_value[0]
                    index = [getattr(sgn, 'value') == cl_value[0] for sgn in all_signs_cls_value].index(True)
                except ValueError as e:
                    print e
                    print "error"

                clusters_output.append(class_value(cl_value[0], all_signs_cls_value[index][1], cl_value[2]))

        clusters_cnt = {}
        for cl in clusters_output:
            if (clusters_cnt.has_key(cl[2])):
                clusters_cnt[cl[2]].append(cl[1])
            else:
                clusters_cnt[cl[2]] = []

        # for k in clusters_cnt.keys():
        #      print clusters_cnt[k]

        sum = 0.0
        for c in clusters_cnt.keys():
            cl_names_in = clusters_cnt[c]
            cl_names_in_unique = unique(cl_names_in)
            max = 0.0
            total = len(cl_names_in)
            # print "total:",total
            for name in cl_names_in_unique:
                names_in_num = cl_names_in.count(name)
                # print names_in_num
                if names_in_num > max:
                    max = names_in_num
            sum += (np.float32(max) / total)

        print sum / len(clusters_cnt)

suffix_len = 17

dm1 = 30
dm2 = 30

for j in range(1,13):
    sign_index = j
    try:
        calc_error()
    except Exception as e:
        print e
        print "error"
