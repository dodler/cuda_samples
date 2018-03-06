from os.path import isfile, join

from os import listdir
import numpy as np

path = "/home/lyan/Documents/image_statistic/output/"

files = [f for f in listdir(path) if isfile(join(path, f))]

print len(files)

for f in files:
	with open(path + f) as f1:
		res = np.float32(f1.readlines())
		if (np.isnan(res).any() or np.isinf(res).any()):
			continue
		print "\""+f+"\","



