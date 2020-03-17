# -*- coding: utf-8 -*-

# This code was used with
#    OS: Ubuntu 18.04LTS
#    Programming language: Python 3.7
#    Deep Learning library: tensorflow-gpu 1.4.1, Keras 2.1.5
#                           CUDA toolkit 8.0, CuDNN v5.1
#    Python libraries: numpy 1.14.2, Pillow 5.0.0
#
# If NeuralNet == "Xception":
#     this code takes about 4 min for training (100 epoches, 320 train/valid)
#     with core i7 6850K, RAM 256GB, NVMe SSD w 3.5" HDD, 1080ti.

import os, keras
import numpy as np
from datetime import datetime
from PIL import Image
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Conv2D
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam

# to select which neuralnetwork to use

#NeuralNet = 'VGG16'       # ILSVRC image classification top-1 accuracy of 0.715
#NeuralNet = 'VGG19'       # ILSVRC image classification top-1 accuracy of 0.727
NeuralNet = 'ResNet50'    # ILSVRC image classification top-1 accuracy of 0.759
#NeuralNet = 'DenseNet201' # ILSVRC image classification top-1 accuracy of 0.770
#NeuralNet = 'InceptionV3' # ILSVRC image classification top-1 accuracy of 0.788
#NeuralNet = 'Xception'    # ILSVRC image classification top-1 accuracy of 0.790
#NeuralNet = 'IncResV2'    # ILSVRC image classification top-1 accuracy of 0.804

# making training data
image_list = []
label_list = []

LAD = np.load("pickup10_2018_LAD_120.npy")/255
LCX = np.load("pickup10_2018_LCX_120.npy")/255
RCA = np.load("pickup10_2018_RCA_120.npy")/255
N = np.load("pickup10_2018_N_120.npy")/255
    #shape: ((X, Y), cycle, caseID, axis)
    #axis = ('s', '2', '3', '4')

LADs = LAD[:, :, :, :, 0].reshape((120, 120, 10, 50, 1))#.tolist() # 0: short axis
LCXs = LCX[:, :, :, :, 0].reshape((120, 120, 10, 50, 1))#.tolist()
RCAs = RCA[:, :, :, :, 0].reshape((120, 120, 10, 50, 1))#.tolist()
Ns = N[:, :, :, :, 0].reshape((120, 120, 10, 50, 1))#.tolist()
print(Ns.shape)

#for tmp in [LADs, LCXs, RCAs, N]:
#    tmp = np.array(tmp).reshape((120, 120, 10, 50, 1))

tmp = np.append(LADs, LCXs, axis = 4)
tmp = np.append(tmp, RCAs, axis = 4)
image_list = np.append(tmp, Ns, axis = 4)
image_list = image_list.reshape((120, 120, 10, 200))
image_list = image_list.transpose((3, 2, 0, 1)) #(200, 10, 120, 120)

zeros150 = np.ones(150).astype('uint8')
ones50   = np.zeros(50).astype('uint8')
label_list = np.append(zeros150, ones50, axis = 0)
label_list = label_list.tolist()


#making neural network
if NeuralNet == 'VGG16':
    print('NeuralNetwork: VGG16.\nILSVRC top-1 accuracy of 0.715')
    DCNN = keras.applications.vgg16.VGG16(
        include_top=True, input_tensor=None, pooling=None, classes=1000)
elif NeuralNet == 'VGG19':
    print('NeuralNetwork: VGG16.\nILSVRC top-1 accuracy of 0.727')
    DCNN = keras.applications.vgg19.VGG19(
        include_top=True, input_tensor=None, pooling=None, classes=1000)
elif NeuralNet == 'ResNet50':
    print('NeuralNetwork: ResNet50.\nILSVRC top-1 accuracy of 0.759')
    DCNN = keras.applications.resnet50.ResNet50(
        include_top=True, input_tensor=None, pooling=None, classes=1000)
elif NeuralNet == 'DenseNet201':
    print('NeuralNetwork: DenseNet201.\nILSVRC top-1 accuracy of 0.770')
    DCNN = keras.applications.densenet.DenseNet201(
        include_top=True, input_tensor=None, pooling=None, classes=1000)
elif NeuralNet == 'InceptionV3':
    print('NeuralNetwork: InceptionV3.\nILSVRC top-1 accuracy of 0.788')
    DCNN = keras.applications.inception_v3.InceptionV3(
        include_top=True, input_tensor=None, pooling=None, classes=1000)
elif NeuralNet == 'Xception':
    print('NeuralNetwork: Xception.\nILSVRC top-1 accuracy of 0.790')
    DCNN = keras.applications.xception.Xception(
        include_top=True, input_tensor=None, pooling=None, classes=1000)
elif NeuralNet == 'IncResV2':
    print('NeuralNetwork: Inception-ResNet-V2.\nILSVRC top-1 accuracy of 0.804')
    DCNN = keras.applications.inception_resnet_v2.InceptionResNetV2(
    include_top=True, input_tensor=None, pooling=None, classes=1000)
else:
    print('error, no neural network.')

opt = Adam(lr = 0.0001)

model = Sequential()
model.add(Dropout(0.5), input_shape=(10, 120, 120)))
model.add(Conv2D(3, (1, 1), activation='relu'))
model.add((DCNN))
model.add(Dropout(0.5))
model.add(Dense(4))
model.add(Activation("softmax"))
model.compile(loss="categorical_crossentropy",
              optimizer=opt, metrics=["accuracy"])

#training
print('training')
model.fit(image_list, label_list,
          epochs=100, batch_size=16, validation_split=0.2)

#saving post-trained model
prefix = datetime.now().strftime("%Y"+"_"+"%m%d"+"_"+"%H%M")
save_name = NeuralNet + '_' + prefix + '.h5'
model.save_weights(save_name)
print('saving post-trained model:', save_name)
print('finished training.')

print('finished: train_DCNN.py')
