#!/usr/bin/env python3

import numpy as np
import pandas as pd
from mvnc import mvncapi as mvnc
#import mnist    # pip3 install mnist
from sklearn.model_selection import train_test_split
import numpy

# For tensorflow
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train = mnist.train_images()
# y_train = mnist.train_labels()

# x_test = mnist.test_images()
# y_test = mnist.test_labels()

data = pd.read_csv('./input/sign_mnist_train.csv')
x = data.iloc[:, 1:].values
y = data.iloc[:, :1].values.flatten()

output_array = np.eye(25)[y]

x_train, x_test, y_train, y_test = train_test_split(x, output_array, test_size=0.33, random_state=42)


x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print(x_train.shape)
# # Prepare test image
test_idx = numpy.random.randint(0, x_train.shape[0])
test_image = x_test[test_idx]

test_image = test_image.astype('float32') / 255.0

# Using NCS Predict
devices = mvnc.EnumerateDevices()
device = mvnc.Device(devices[0])
device.OpenDevice()

with open("graph", mode='rb') as f:
    graphfile = f.read()

graph = device.AllocateGraph(graphfile)

graph.LoadTensor(test_image.astype('float32'), 'user object')

output, userobj = graph.GetResult()

graph.DeallocateGraph()
device.CloseDevice()

print("NCS", output, output.argmax())
print("Correct", y_test[test_idx])
