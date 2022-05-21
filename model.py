import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
from numpy.random import seed
seed(1607)
from tensorflow import set_random_seed
set_random_seed(1607)

from keras.models import Model, Sequential, model_from_json
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from keras.layers import Input, Dense, Layer, Reshape, ReLU
from keras.layers import Dropout, Lambda, Multiply, Add, Subtract, Concatenate
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose, MaxPooling2D
import keras.backend as K

import numpy as np

latent_dim = 1024
latent_shape = (16, 16, 4)

H, W, C = 128, 128, 1
img_shape = (H, W, C)
key_shape = (latent_dim, )

def arr_img(img):
    return np.array(img*255.0, dtype='uint8')

def encoder_model():
    i_input = Input(shape=img_shape)
    
    e_conv_0 = Sequential(name='e_conv_0')
    e_conv_0.add(Conv2D(64, (3, 3), padding='same', input_shape=img_shape))
    e_conv_0.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    enc = e_conv_0(i_input)
    
    e_conv_1 = Sequential(name='e_conv_1')
    e_conv_1.add(Conv2D(64, (3, 3), padding='same', strides=2, input_shape=(128, 128, 64)))
    e_conv_1.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    enc = e_conv_1(enc)
    
    e_conv_2 = Sequential(name='e_conv_2')
    e_conv_2.add(Conv2D(64, (3, 3), padding='same', strides=2, input_shape=(64, 64, 64)))
    e_conv_2.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    enc = e_conv_2(enc)
    
    e_conv_3 = Sequential(name='e_conv_3')
    e_conv_3.add(Conv2D(64, (3, 3), padding='same', strides=2, input_shape=(32, 32, 64)))
    e_conv_3.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    enc = e_conv_3(enc)
    
    e_conv_4 = Sequential(name='e_conv_4')
    e_conv_4.add(Conv2D(4, (3, 3), padding='same', input_shape=(16, 16, 64)))
    e_conv_4.add(Conv2D(4, (3, 3), activation='relu', padding='same'))
    enc = e_conv_4(enc)
                
    enc = Flatten()(enc)
    return Model(i_input, enc, name='encoder')

encoder = encoder_model()

def decoder_model():
    l_input = Input(shape=(latent_dim, ))
    dec = Reshape(latent_shape)(l_input)
    
    d_dconv_0 = Sequential(name='d_dconv_0')
    d_dconv_0.add(Conv2DTranspose(64, (3, 3), padding='same', input_shape=latent_shape))
    d_dconv_0.add(Conv2DTranspose(64, (3, 3), activation='relu', padding='same'))
    dec = d_dconv_0(dec)
        
    d_dconv_1 = Sequential(name='d_dconv_1')
    d_dconv_1.add(Conv2DTranspose(64, (3, 3), padding='same', strides=2, input_shape=(16, 16, 64)))
    d_dconv_1.add(Conv2DTranspose(64, (3, 3), activation='relu', padding='same'))
    dec = d_dconv_1(dec)
    
    d_dconv_2 = Sequential(name='d_dconv_2')
    d_dconv_2.add(Conv2DTranspose(64, (3, 3), padding='same', strides=2, input_shape=(32, 32, 64)))
    d_dconv_2.add(Conv2DTranspose(64, (3, 3), activation='relu', padding='same'))
    dec = d_dconv_2(dec)
    
    d_dconv_3 = Sequential(name='d_dconv_3')
    d_dconv_3.add(Conv2DTranspose(64, (3, 3), padding='same', strides=2, input_shape=(64, 64, 64)))
    d_dconv_3.add(Conv2DTranspose(64, (3, 3), activation='relu', padding='same'))
    dec = d_dconv_3(dec)
    
    dec = Conv2D(C, (3, 3), padding='same', activation='sigmoid', name='d_dconv_4')(dec)
   
    return Model(l_input, dec, name='decoder')

decoder = decoder_model()

def build_ecan():
    e_input = Input(shape=(latent_dim, ))
    k_input = Input(shape=(latent_dim, ))
    ecan = Add()([e_input, k_input])
    ecan = Dense(latent_dim, use_bias=True, activation=None)(ecan)
    return Model([e_input, k_input], ecan, name='enc_can')

ec = build_ecan()

def build_dcan():
    d_input = Input(shape=(latent_dim, ))
    k_input = Input(shape=(latent_dim, ))
    dcan = Dense(latent_dim, use_bias=True, activation=None)(d_input)
    dcan = Subtract()([dcan, k_input])
    return Model([d_input, k_input], dcan, name='dec_can')

dc = build_dcan()

def build_complete():
    e_input = Input(shape=img_shape)
    k_input = Input(shape=(latent_dim, ), name='key')
    ae = encoder(e_input)
    ae = ec([ae, k_input])
    ae = dc([ae, k_input])
    ae = decoder(ae)
    return Model([e_input, k_input], ae)
    
ae = build_complete()

optimizer = Adam(1e-4, decay=1e-6)
ae.compile(loss='mse', optimizer=optimizer)
# ae.load_weights('dcae_v6.h5')
ae.summary()

def get_latent(img):
    img = img.reshape(1, img_shape[0], img_shape[1], 1)
    elat = encoder.predict(img)
    return elat

def get_latent_key(elat, key):
    key = key.reshape(1, key_shape[0])
    lat = ec.predict([elat, key])
    return lat
