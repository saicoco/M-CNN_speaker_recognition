# -*- coding: utf-8 -*-
# author = sai
from keras.layers import Convolution2D, Dense, BatchNormalization, Activation, Flatten, merge, Input, Merge, MaxPooling2D
from keras.models import Sequential, Model
from keras.optimizers import SGD

def conv2d_bn(inputs, n_filters, width, heigth, border_mode='same', subsample=(1, 1), bn=True):
    x = Convolution2D(nb_filter=n_filters, nb_row=width, nb_col=heigth, border_mode=border_mode, subsample=subsample)(inputs)
    if bn:
        x = BatchNormalization()(x)
    x = Activation('sigmoid')(x)
    return x

def model_face_alone(nclasses=6):
    face_model = Sequential()
    # conv1
    face_model.add(Convolution2D(48, 15, 1, border_mode='same', input_shape=(3, 50, 50)))
    face_model.add(Convolution2D(48, 1, 15, border_mode='same'))
    face_model.add(BatchNormalization(axis=1))
    face_model.add(Activation('sigmoid'))
    face_model.add(Convolution2D(48, 3, 3, border_mode='same', subsample=(2, 2)))
    face_model.add(BatchNormalization(axis=1))
    face_model.add(Activation('sigmoid'))

    # conv2
    face_model.add(Convolution2D(256, 3, 3, border_mode='same'))
    face_model.add(Convolution2D(256, 3, 3, border_mode='same'))
    face_model.add(BatchNormalization(axis=1))
    face_model.add(Activation('sigmoid'))
    face_model.add(Convolution2D(256, 3, 3, border_mode='same', subsample=(2, 2)))
    face_model.add(BatchNormalization(axis=1))
    face_model.add(Activation('sigmoid'))
    #face_model.add(Convolution2D(1024, 7, 7, border_mode='same', subsample=(7, 7)))
    face_model.add(Convolution2D(1024, 1, 7, border_mode='same'))
    face_model.add(Convolution2D(1024, 7, 1, border_mode='same'))
    face_model.add(BatchNormalization(axis=1))
    face_model.add(Activation('sigmoid'))

    # FC
    face_model.add(Flatten())
    face_model.add(Dense(1024))
    face_model.add(BatchNormalization(axis=1))
    face_model.add(Activation('sigmoid'))
    face_model.add(Dense(nclasses))
    face_model.add(BatchNormalization(axis=1))
    face_model.add(Activation('softmax'))

    sgd = SGD(lr=0.1, decay=(5e-7) / 10., nesterov=True, momentum=0.9)
    face_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return face_model

def model_face_audio(nclasses=6):
    face_model = Sequential()
    # conv1
    face_model.add(Convolution2D(48, 15, 15, border_mode='same', input_shape=(3, 50, 50)))
    face_model.add(BatchNormalization(axis=1))
    face_model.add(Activation('sigmoid'))
    face_model.add(Convolution2D(48, 3, 3, border_mode='same', subsample=(2, 2)))
    face_model.add(BatchNormalization(axis=1))
    face_model.add(Activation('sigmoid'))

    # conv2
    face_model.add(Convolution2D(256, 3, 3, border_mode='same'))
    face_model.add(Convolution2D(256, 3, 3, border_mode='same'))
    face_model.add(BatchNormalization(axis=1))
    face_model.add(Activation('sigmoid'))
    face_model.add(Convolution2D(256, 3, 3, border_mode='same', subsample=(2, 2)))
    face_model.add(BatchNormalization(axis=1))
    face_model.add(Activation('sigmoid'))
    face_model.add(Convolution2D(1024, 7, 7, border_mode='same', subsample=(7, 7)))
    face_model.add(BatchNormalization(axis=1))
    face_model.add(Activation('sigmoid'))
    face_model.add(Flatten())

    # audio
    audio_model = Sequential()
    audio_model.add(Dense(375, input_dim=375))
    # merge modal
    merge_modal = Sequential()
    merge_modal.add(Merge([face_model, audio_model], mode='concat'))
    merge_modal.add(Dense(1024))
    merge_modal.add(BatchNormalization())
    merge_modal.add(Activation('sigmoid'))
    merge_modal.add(Dense(nclasses))
    merge_modal.add(BatchNormalization())
    merge_modal.add(Activation('softmax'))

    sgd = SGD(lr=0.1, decay=(5e-7) / 10., nesterov=True, momentum=0.9)
    merge_modal.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return face_model, audio_model,merge_modal

def model_v1(n_classes=6):
    input = Input(shape=(3, 50, 50))
    # branch 1
    x = conv2d_bn(inputs=input, n_filters=48, width=15, heigth=15, border_mode='valid')

    # mixed 208 36*36
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    x = merge([branch1x1, branch5x5, branch3x3dbl], mode='concat', concat_axis=1)

    # mixde_688 18*18
    branch3x3 = conv2d_bn(x, 384, 3, 3, border_mode='valid', subsample=(2, 2))

    branch1x1db1 = conv2d_bn(x, 64, 1, 1)
    branch3x3db1 = conv2d_bn(branch1x1db1, 96, 3, 3)
    branch3x3db = conv2d_bn(branch3x3db1, 96, 3, 3, subsample=(2, 2), border_mode='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = merge([branch3x3, branch3x3db, branch_pool], mode='concat', concat_axis=1)

    # mixed 14*14
    branch3x3 = conv2d_bn(x, 384, 3, 3, border_mode='valid')
    branch3x3 = conv2d_bn(branch3x3, 384, 3, 3, border_mode='valid')

    branch1x1db1 = conv2d_bn(x, 64, 1, 1)
    branch3x3db1 = conv2d_bn(branch1x1db1, 96, 5, 5, border_mode='valid')
    x = merge([branch3x3, branch3x3db1], mode='concat', concat_axis=1)

    # mixed 7*7
    branch3x3 = conv2d_bn(x, 384, 3, 3, border_mode='valid', subsample=(2, 2))

    branch1x1db1 = conv2d_bn(x, 64, 1, 1)
    branch3x3db1 = conv2d_bn(branch1x1db1, 96, 3, 3)
    branch3x3db = conv2d_bn(branch3x3db1, 96, 3, 3, subsample=(2, 2), border_mode='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = merge([branch3x3, branch3x3db, branch_pool], mode='concat', concat_axis=1)

    # output1`
    output1 = conv2d_bn(x, 1024, 1, 1)
    output1 = conv2d_bn(output1, 1024, 7, 7)
    output1 = Flatten()(output1)
    output1 = Dense(1024)(output1)
    output1 = BatchNormalization()(output1)
    output1 = Activation('sigmoid')(output1)
    output1 = Dense(n_classes)(output1)
    output1 = BatchNormalization()(output1)
    aux = Activation('softmax')(output1)
    # 3*3
    branch3x3 = conv2d_bn(x, 384, 3, 3, border_mode='valid', subsample=(2, 2))

    branch1x1db1 = conv2d_bn(x, 64, 1, 1)
    branch3x3db1 = conv2d_bn(branch1x1db1, 96, 3, 3)
    branch3x3db = conv2d_bn(branch3x3db1, 96, 3, 3, subsample=(2, 2), border_mode='valid')

    branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = merge([branch3x3, branch3x3db, branch_pool], mode='concat', concat_axis=1)
    # 1*1
    branch3x3 = conv2d_bn(x, 384, 3, 3)

    branch1x1db1 = conv2d_bn(x, 64, 1, 1)
    branch3x3db1 = conv2d_bn(branch1x1db1, 96, 3, 3)
    x = merge([branch3x3, branch3x3db1], mode='concat', concat_axis=1)

    x = Flatten()(x)
    # FC
    fc1 = Dense(1024)(x)
    bn_fc1 = BatchNormalization()(fc1)
    ac_fc1 = Activation('sigmoid')(bn_fc1)
    bn_softmax = Dense(n_classes)(ac_fc1)
    bn_soft = BatchNormalization()(bn_softmax)
    output = Activation('softmax')(bn_soft)
    model = Model(input=input, output=[aux, output])
    sgd = SGD(lr=0.1, decay=(5e-7) / 10., nesterov=True, momentum=0.9)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def model_v2(n_classes=6):
    x = Input(shape=(3, 50 ,50))
    # face model
    conv15x15 = Convolution2D(48, 1, 15)(x)
    conv15x15 = Convolution2D(48, 15, 1)(x)
    bn15x15 = BatchNormalization(axis=1)(conv15x15)
    ac15x15 = Activation('sigmoid')(bn15x15)
    mxpool = MaxPooling2D()(ac15x15)

    conv3x3 = conv2d_bn(mxpool, 256, 3, 3, border_mode='valid')
    conv3x3 = conv2d_bn(conv3x3, 256, 3, 3, border_mode='valid')
    mxpool = MaxPooling2D()(conv3x3)

    conv7x7 = Convolution2D(1024, 1, 7)(mxpool)
    conv7x7 = Convolution2D(1024, 7, 1)(conv7x7)
    bn7x7 = BatchNormalization(axis=1)(conv7x7)
    ac7x7 = Activation('sigmoid')(bn7x7)

    flaten = Flatten()(ac7x7)
    fc1024 = Dense(1024)(flaten)
    bn1024 = BatchNormalization()(fc1024)
    ac1024 = Activation('sigmoid')(bn1024)

    soft6 = Dense(n_classes)(ac1024)
    bn_fa = BatchNormalization()(soft6)
    ac_fa = Activation('softmax')(bn_fa)

    y = Input(shape=(375,))
    fc357 = Dense(357)(y)
    bn357 = BatchNormalization()(fc357)
    ac357 = Activation('sigmoid')(bn357)
    fc_ad = Dense(6)(ac357)
    bn_ad = BatchNormalization()(fc_ad)
    ac_ad = Activation('softmax')(bn_ad)

    v_a = merge([ac_fa, ac_ad], mode='sum')

    fc_me = Dense(6)(v_a)
    bn_m = BatchNormalization()(fc_me)
    ac_me = Activation('softmax')(bn_m)

    merge_modal = Model(input=[x, y], output=[ac_me])
    sgd = SGD(lr=0.1, decay=(5e-7) / 10., nesterov=True, momentum=0.9)
    merge_modal.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return merge_modal
def model_v3(n_classes=6):
    x = Input(shape=(3, 50 ,50))
    # face model
    conv15x15 = Convolution2D(48, 1, 15)(x)
    conv15x15 = Convolution2D(48, 15, 1)(x)
    bn15x15 = BatchNormalization(axis=1)(conv15x15)
    ac15x15 = Activation('sigmoid')(bn15x15)
    mxpool = MaxPooling2D()(ac15x15)

    conv3x3 = conv2d_bn(mxpool, 256, 3, 3, border_mode='valid')
    conv3x3 = conv2d_bn(conv3x3, 256, 3, 3, border_mode='valid')
    mxpool = MaxPooling2D()(conv3x3)

    conv7x7 = Convolution2D(1024, 1, 7)(mxpool)
    conv7x7 = Convolution2D(1024, 7, 1)(conv7x7)
    bn7x7 = BatchNormalization(axis=1)(conv7x7)
    ac7x7 = Activation('sigmoid')(bn7x7)

    flaten = Flatten()(ac7x7)
    fc1024 = Dense(1024)(flaten)
    bn1024 = BatchNormalization()(fc1024)
    ac1024 = Activation('sigmoid')(bn1024)


    y = Input(shape=(375,))
    fc357 = Dense(357)(y)
    bn357 = BatchNormalization()(fc357)
    ac357 = Activation('sigmoid')(bn357)

    v_a = merge([ac1024, ac357], mode='concat')
    fc_me = Dense(6)(v_a)
    bn_m = BatchNormalization()(fc_me)
    ac_me = Activation('softmax')(bn_m)
    merge_modal = Model(input=[x, y], output=[ac_me])
    sgd = SGD(lr=0.1, decay=(5e-7) / 10., nesterov=True, momentum=0.9)
    merge_modal.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return merge_modal
