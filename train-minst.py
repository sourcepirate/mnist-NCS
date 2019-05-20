
# coding: utf-8

# In[1]:


import pandas as pd
from keras import layers, models
from keras.models import load_model
from keras.datasets import mnist
from keras.utils import to_categorical
from keras import backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings
import os
import numpy as np
from sklearn.model_selection import train_test_split



# In[2]:
warnings.filterwarnings('ignore')
print(os.listdir("input/"))

data = pd.read_csv('./input/sign_mnist_train.csv')
print(data.shape)

print(f"Number of images: {len(data.iloc[:, :1].values)}")
print(f"Pixel square of : {data.shape[1]}")



def next_batch(batch_size, data, labels):
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[: batch_size]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def display_images(data):
    x, y = data
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace = 0.5, wspace = 0.5)
    for i, ax in enumerate(axes.flat):
        ax.imshow(x[i].reshape(28, 28), cmap = 'binary')
        ax.set_xlabel(chr(y[i] + 65))
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

x = data.iloc[:, 1:].values
y = data.iloc[:, :1].values.flatten()

img_arr = np.asarray(x[0].reshape(28, 28))
plt.imshow(img_arr, cmap="binary")

display_images(next_batch(9, x, y))

output_array = np.eye(25)[y]





x_train, x_test, y_train, y_test = train_test_split(x, output_array, test_size=0.33, random_state=42)


# In[3]:


x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)


# In[4]:



# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)


print(y_train[0])
# In[5]:


model = models.Sequential()
model.add(layers.Conv2D(16, 3, activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPool2D())
model.add(layers.Conv2D(32, 3, activation='relu'))
model.add(layers.MaxPool2D())
model.add(layers.Conv2D(64, 3, activation='relu'))
model.add(layers.MaxPool2D())
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(25, activation='softmax'))
model.summary()


# In[6]:


model.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')


# In[7]:


history = model.fit(x_train, y_train, epochs=2, batch_size=128)

# In[8]:


print(model.evaluate(x_test, y_test))


# In[9]:


# model 與 weights 分別儲存
with open("metadata.json", "w") as file:
    file.write(model.to_json())
model.save_weights("weights.h5")


# In[10]:


# model 與 weights存在同一檔案中,for convert-mnist-only-h5.ipynb
model.save('model.h5')

