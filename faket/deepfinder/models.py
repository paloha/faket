# =============================================================================================
# DeepFinder - a deep learning approach to localize macromolecules in cryo electron tomograms
# =============================================================================================
# Copyright (C) Inria,  Emmanuel Moebel, Charles Kervrann, All Rights Reserved, 2015-2021, v1.0
# License: GPL v3.0. See <https://www.gnu.org/licenses/>
# =============================================================================================

from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.layers import Conv3D, MaxPooling3D, UpSampling3D
from tensorflow.keras.models import Model

import tensorflow as tf

def my_model(dim_in, Ncl):
    tf.random.set_seed(1234)
    
    input = Input(shape=(dim_in,dim_in,dim_in,1))
    
    x    = Conv3D(32, (3,3,3), padding='same', activation='relu')(input)
    high = Conv3D(32, (3,3,3), padding='same', activation='relu')(x)
    
    x = MaxPooling3D((2,2,2), strides=None)(high)
    
    x   = Conv3D(48, (3,3,3), padding='same', activation='relu')(x)
    mid = Conv3D(48, (3,3,3), padding='same', activation='relu')(x)
    
    x = MaxPooling3D((2,2,2), strides=None)(mid)
    
    x = Conv3D(64, (3,3,3), padding='same', activation='relu')(x)
    x = Conv3D(64, (3,3,3), padding='same', activation='relu')(x)
    x = Conv3D(64, (3,3,3), padding='same', activation='relu')(x)
    x = Conv3D(64, (3,3,3), padding='same', activation='relu')(x)

    x = UpSampling3D(size=(2,2,2), data_format='channels_last')(x)
    x = Conv3D(64, (2,2,2), padding='same', activation='relu')(x)
    
    x = concatenate([x, mid])
    x   = Conv3D(48, (3,3,3), padding='same', activation='relu')(x)
    x   = Conv3D(48, (3,3,3), padding='same', activation='relu')(x)
    
    x = UpSampling3D(size=(2,2,2), data_format='channels_last')(x)
    x = Conv3D(48, (2,2,2), padding='same', activation='relu')(x)
    
    x = concatenate([x, high])
    x = Conv3D(32, (3,3,3), padding='same', activation='relu')(x)
    x = Conv3D(32, (3,3,3), padding='same', activation='relu')(x)
    output = Conv3D(Ncl, (1,1,1), padding='same', activation='softmax')(x)
    
    model = Model(input, output)
    return model