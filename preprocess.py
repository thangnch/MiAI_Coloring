import cv2
import os
import glob
import pickle
from tqdm import tqdm
import numpy as np
import sys

def split_hsv_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img_hsv[:, :, :2], img_hsv[:, :, 2]


X = []
y = []
count = 0
nb_file = 0
for path, dirs, files in os.walk('data_raw'):
    for f in tqdm(glob.iglob(os.path.join(path, '*.jpg'))):
        count += 1
        print(count)
        hs, v = split_hsv_img(f)
        X.append(v)
        y.append(hs)

        if count % 10000 == 0:
            with open('data_processed/X_' + str(count // 10000) + '.pkl', 'wb') as f:
                pickle.dump(X, f)
            with open('data_processed/y_' + str(count // 10000) + '.pkl', 'wb') as f:
                pickle.dump(y, f)
            print(count // 10000, np.shape(X), np.shape(y))
            nb_file = nb_file + 1
            if nb_file>=10:
                print("Done")
                sys.exit(0)
            del X, y
            X = []
            y = []