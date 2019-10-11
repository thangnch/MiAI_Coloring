from tensorflow.keras.models import model_from_json
from matplotlib import pyplot as plt
import cv2
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras import applications
from random import randint
import cv2
import numpy as np
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

'''
json_file = open('models/model_15_1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
'''

resnet = applications.resnet50.ResNet50(weights=None, classes=365)
resnet.summary()
x = resnet.output
model_tmp = Model(inputs = resnet.input, outputs = x)
model_tmp.summary()

#layer_3, layer_7, layer_13, layer_16 = model_tmp.get_layer('conv1_relu').output,\
#                                       model_tmp.get_layer('conv2_block3_out').output, \
#                                       model_tmp.get_layer('conv3_block4_out').output, \
#                                       model_tmp.get_layer('conv4_block6_out').output

layer_3, layer_7, layer_13, layer_16 = model_tmp.get_layer('activation_9').output, model_tmp.get_layer('activation_21').output, model_tmp.get_layer('activation_39').output, model_tmp.get_layer('activation_48').output

fcn1 = Conv2D(filters=2 , kernel_size=1, name='fcn1')(layer_16)

fcn2 = Conv2DTranspose(filters=layer_13.get_shape().as_list()[-1] , kernel_size=4, strides=2, padding='same', name='fcn2')(fcn1)
fcn2_skip_connected = Add(name="fcn2_plus_layer13")([fcn2, layer_13])

fcn3 = Conv2DTranspose(filters=layer_7.get_shape().as_list()[-1], kernel_size=4, strides=2, padding='same', name="fcn3")(fcn2_skip_connected)
fcn3_skip_connected = Add(name="fcn3_plus_layer_7")([fcn3, layer_7])

fcn4 = Conv2DTranspose(filters=layer_3.get_shape().as_list()[-1], kernel_size=4, strides=2, padding='same', name="fcn4")(fcn3_skip_connected)
fcn4_skip_connected = Add(name="fcn4_plus_layer_3")([fcn4, layer_3])

# Upsample again
fcn5 = Conv2DTranspose(filters=2, kernel_size=16, strides=(4, 4), padding='same', name="fcn5")(fcn4_skip_connected)
relu255 = ReLU(max_value=255) (fcn5)

model = Model(inputs = resnet.input, outputs = relu255)

# load weights into new model
model.load_weights("models/model_12_1.h5")
print("Loaded model from disk")


import pickle

with open('data_processed/X_10.pkl', 'rb') as f:
    X = pickle.load(f)
with open('data_processed/y_10.pkl', 'rb') as f:
    y = pickle.load(f)


def regenerate_img(gray, hs):
    img = np.zeros((224, 224, 3))
    img[:, :, 2] = gray  # V
    img[:, :, 0] = hs[:, :, 0]  # H
    img[:, :, 1] = hs[:, :, 1]  # S

    img = np.array(img, np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)  # Convert to RGB

    return img

def predict_hs(gray):
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return model.predict(np.array([img]))[0]

index = np.random.choice(a=len(X), size=20)

grays = []
predicts = []
originals = []
for i in index:
    grays.append(X[i])
    predicts.append(regenerate_img(X[i], predict_hs(X[i])))
    originals.append(regenerate_img(X[i], y[i]))


print(len(grays))

fig = plt.figure(figsize=(15, 15/3*len(grays)))
columns = 3
rows = len(grays)
show = []

for idx in range(2):
    cv2.imshow("Gray" + str(idx),cv2.cvtColor(grays[idx],cv2.COLOR_RGB2BGR))
    cv2.imshow("Painted" + str(idx),cv2.cvtColor(predicts[idx],cv2.COLOR_RGB2BGR))
    cv2.imshow("Fact" + str(idx), cv2.cvtColor(originals[idx],cv2.COLOR_RGB2BGR))
cv2.waitKey()

print("Done")