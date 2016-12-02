
# coding: utf-8

# In[7]:

from __future__ import print_function
import numpy as np
import pandas as pd
import scipy
from sklearn.preprocessing import MinMaxScaler
import glob
import os
import matplotlib.pyplot as plt
import pickle
# %matplotlib inline


# In[8]:

from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from keras.models import Sequential
#from keras.initializations import norRemal, identity
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.optimizers import RMSprop, Adadelta
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Activation, TimeDistributedDense, Dropout, Reshape, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.models import model_from_json

from keras.models import Sequential
from keras.models import model_from_json
from sklearn.metrics import log_loss
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
# from joblib import Parallel, delayed


# In[9]:

min_max_scaler = MinMaxScaler()


# In[10]:

# config
rows            = 50
cols            = 70


# In[11]:

def load_csv(n_rows=None):

    file = pd.read_csv("../capture_inferno_orginal.csv")
    file.columns = ['index', 'steer', 'gear', 'accel', 'brake', 'clutch', 'speed']
    file = file[['index', 'speed']]
    file['index'] = file['index'].apply(lambda x: str(x) + ".png")
    return(file)


# In[12]:

def get_im_cv(path, img_rows, img_cols, color_type=1):
    # Load as grayscale
    if color_type == 1:
        img = scipy.misc.imread(path, 0)
    elif color_type == 3:
        img = scipy.misc.imread(path, mode='RGB')
    # Reduce size
    resized = scipy.misc.imresize(img, (img_cols, img_rows))
    return resized


# In[13]:

def load_images(raw_images, dataset_csv):
    dataset = dict(zip(dataset_csv['index'], dataset_csv['speed']))
    images = []
    speeds = []
    ids = []
    for raw_image in raw_images:
        try:
            img = get_im_cv(raw_image, cols, rows, 3)
            raw_image_name = raw_image.split('/')[-1]
            speed = dataset[raw_image_name]
            images.append(img)
            ids.append(raw_image_name)
            speeds.append(speed)
        except:
            pass

    return(np.array(images), np.array(speeds), np.array(ids))
        


# In[14]:

def create_sequence_data(X_data, y, numToAdd):
    X_tmp = []
    y_tmp = []
    data_length = len(X_data)
    for i in range(0, data_length):

        if (i + numToAdd) >= data_length:
            break
        X_tmp.append(X_data[i:i+numToAdd])
        y_tmp.append(y[i+numToAdd-1])
    return(np.array(X_tmp), np.array(y_tmp))


# In[26]:

def split_validation_set(train, target, maxToAdd, test_size = 0.2):
    random_state = 51
    target = target.reshape(-1, 1)
    target = min_max_scaler.fit_transform(target)
    thresh = int(len(train) * 0.8)
    X_tmp = train[:thresh]
    X_test = train[thresh:]
    y_tmp = target[:thresh]
    y_test = target[thresh:]
#     X_tmp, X_test, y_tmp, y_test = train_test_split(train, target, test_size=test_size, random_state=random_state)
    
#     X_tmp, X_test, y_tmp, y_test = train_test_split(train, target, test_size=test_size, random_state=random_state)
#     X_train, X_validation, y_train, y_validation = train_test_split(X_tmp, y_tmp, test_size=test_size, random_state=random_state)
    thresh2 = int(len(X_tmp) * 0.8)
    X_train = X_tmp[:thresh2]
    X_validation = X_tmp[thresh2:]
    y_train = y_tmp[:thresh2]
    y_validation = y_tmp[thresh2:]
    
    X_train, y_train = create_sequence_data(X_train, y_train, 20)
    X_validation, y_validation = create_sequence_data(X_validation, y_validation, 20)
    X_test, y_test = create_sequence_data(X_test, y_test, 20)
    
    return(X_train, X_test, y_train, y_test, X_validation, y_validation)


# In[16]:

def create_model(maxToAdd,width, height):
    model = Sequential()
    
    model.add(TimeDistributed(Convolution2D(8, 4, 4, border_mode='valid'), input_shape=(maxToAdd,width, height,1)))
    model.add(Activation('relu'))
    model.add(TimeDistributed(Convolution2D(16, 3, 3, border_mode='valid')))
    model.add(Activation('relu'))
    model.add(TimeDistributed(Convolution2D(24, 5, 5, border_mode='valid'), input_shape=(maxToAdd,width,height, 1)))
    model.add(Activation('relu'))
    
#     model.add(TimeDistributed(Convolution2D(36, 5, 5, border_mode='valid')))
#     model.add(Activation('relu'))
    
#     model.add(TimeDistributed(Convolution2D(64, 5, 5, border_mode='valid')))
#     model.add(Activation('relu'))
    
#     model.add(TimeDistributed(Convolution2D(64, 5, 5, border_mode='valid')))
#     model.add(Activation('relu'))
    
# #     model.add(Reshape((maxToAdd,np.prod(model.output_shape[-3:])))) #this line updated to work with keras 1.0.2
#     model.add(Reshape((maxToAdd,np.prod(model.output_shape[-3:]))))
    model.add(TimeDistributed(Flatten()))
    model.add(Activation('relu'))
#     model.add(GRU(output_dim=500,return_sequences=True))
    model.add(GRU(output_dim=100,return_sequences=True))
    model.add(GRU(output_dim=50,return_sequences=False))
    model.add(Dropout(.2))
    model.add(Dense(1))

    rmsprop = RMSprop()
    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())
    return(model)


# In[18]:

raw_images = glob.glob("../images/*.png")[:40000]
raw_images = sorted(raw_images)
raw_images = np.array(raw_images)
raw_images = raw_images[::2]
raw_images = raw_images[100:]


# In[19]:

csv_file = load_csv()


# In[20]:

raw_images.shape


# In[21]:

# for i in range(500, 520):
#     plt.imshow(cv2.imread(tmp_images[i]))
#     plt.show()


# In[22]:

print('data loading...')
images, speeds, ids = load_images(raw_images, csv_file)


# In[23]:

images = images[:, :, :, :1]
images = images.astype('float32')
images /= 255


# In[24]:

batch_size      = 32
nb_epochs       = 20
examplesPer     = 500
maxToAdd        = 20
hidden_units    = 200



# In[27]:

print('data splitting...')
X_train, X_test, y_train, y_test, X_validation, y_validation = split_validation_set(images, speeds, 20, test_size = 0.2)


# In[29]:

filepath="weights/weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss',verbose=1, save_best_only=True, mode='min')
logger = CSVLogger("overview.csv", separator=',', append=True)
callbacks_list = [checkpoint, logger]


# In[30]:

print(X_train.shape)
print(X_test.shape)
print(X_validation.shape)


# In[31]:

print("model is getting ready...")
model = create_model(maxToAdd, rows, cols)
model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=50, nb_epoch= 50, callbacks=callbacks_list)
# model.fit(X_train, y_train, validation_split=0.2)


# In[ ]:



