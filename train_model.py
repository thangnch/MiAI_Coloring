import pickle
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import applications
from random import randint
import cv2
import numpy as np
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint


def load_data(i):
    with open('data_processed/X_' + str(i) + '.pkl', 'rb') as f:
        images = pickle.load(f)

    with open('data_processed/y_' + str(i) + '.pkl', 'rb') as f:
        gt_images = pickle.load(f)

    return images, gt_images

images_val, gt_images_val = load_data(10)

resnet = applications.resnet50.ResNet50(weights=None, classes=365)
x = resnet.output
model_tmp = Model(inputs = resnet.input, outputs = x)

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


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


model.compile(loss=root_mean_squared_error, optimizer='rmsprop')


def preprocessing_input_output(img, gt):
    img_n = img.copy()
    img_n = cv2.cvtColor(img_n, cv2.COLOR_GRAY2BGR)

    return img_n, gt


def image_batch_generator(images, gt_images, batch_size):
    while True:
        batch_paths = np.random.choice(a=len(images), size=batch_size)
        input = []
        output = []

        for i in batch_paths:
            in_p, out_p = preprocessing_input_output(images[i], gt_images[i])
            input.append(in_p)
            output.append(out_p)

        input = np.array(input)
        output = np.array(output)
        # output = np.array(np.expand_dims(output, -1))
        yield (input, output)

filepath = 'models/model_12_1.h5'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
callbacks_list = [checkpoint]

model.load_weights(filepath)

batch_size = 32
nb_epochs = 150

for i in range(0, nb_epochs):
    i = np.random.randint(1, 9)
    images, gt_images = load_data(i)
    model.fit_generator(generator=image_batch_generator(images, gt_images, batch_size),
                       steps_per_epoch=len(images)//batch_size,
                       epochs=5,
                       verbose=1,
                       validation_data=image_batch_generator(images_val, gt_images_val, batch_size),
                       validation_steps=len(images_val)//batch_size,
                       callbacks=callbacks_list)
    del images, gt_images

model_json = model.to_json()
with open("models/model_15_1.json", "w") as json_file:
    json_file.write(model_json)