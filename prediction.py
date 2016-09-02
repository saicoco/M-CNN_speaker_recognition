# -*- coding: utf-8 -*-
# author = sai

import cPickle, model, model_1
from keras.utils.np_utils import to_categorical
import numpy as np
import data_flow
import logging
from sklearn.metrics import label_ranking_average_precision_score
from keras.callbacks import ModelCheckpoint
train_num = 89833
test_num = 26979
batch_size = 250
n_batches = test_num//batch_size
def predict_image():
    img_val_gene = data_flow.image_generator(batch_size=batch_size, train=False)
    img_gen = data_flow.image_generator(batch_size=batch_size)
    face_model = model.model_face_alone(6)
    face_model.load_weights('weights/face_model.hdf5')
    # filepath = 'weights/weights_face.{epoch:02d}-{val_loss:.2f}.hdf5'
    # callbacks = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=0, save_best_only=True,
                                # save_weights_only=False, mode='auto')
    # face_model.load_weights('./model/friend/face_alone_weights.61-0.2502.h5')
    # face_model.fit_generator(img_gen, samples_per_epoch=train_num, nb_epoch=20, callbacks=[callbacks],validation_data=img_val_gene, nb_val_samples=test_num)
    # score = face_model.evaluate_generator(img_val_gene, val_samples=test_num)
    # print score
    truth = []
    predition = []
    for i in xrange(n_batches):
        imgs, labels = img_val_gene.next()
        truth.append(labels)
        predition.append(face_model.predict(imgs, verbose=1))
    truth = np.concatenate(truth)
    predition = np.concatenate(predition)
    truth = to_categorical(truth, nb_classes=6)
    score = label_ranking_average_precision_score(truth, predition)
    logging.basicConfig(level=logging.INFO)
    logging.info('trueth shape:{}, prediction shape:{}, score:{}'.format(truth.shape, predition.shape, score))
    # results = {'y':truth, 'y_pred':predition}
    # with open('facemodel_score.pkl', 'w') as f:
    #     cPickle.dump(results, f, protocol=1)
def prediction_level(mod):
    av_gen = data_flow.av_generator(batch_size=batch_size, train=False)
    # av_gen_train = data_flow.av_generator(batch_size=batch_size)
    #
    # filepath = 'weights/weights_feature_level_2.{epoch:02d}-{val_loss:.2f}.hdf5'
    # callbacks = ModelCheckpoint(filepath=filepath, monitor='val_loss', verbose=0, save_best_only=False,
    # save_weights_only=False, mode='auto')
    # # mod.load_weights('./model/friend/face_alone_weights.61-0.2502.h5')
    # mod.fit_generator(av_gen_train, samples_per_epoch=train_num, nb_epoch=20, callbacks=[callbacks],validation_data=av_gen, nb_val_samples=test_num)
    # score = mod.evaluate_generator(av_gen, val_samples=test_num)
    # print score

    truth = []
    predition = []
    for i in xrange(n_batches):
        imgs, labels = av_gen.next()
        truth.append(labels)
        predition.append(mod.predict(imgs, verbose=1))
        if i == 1:
            print predition[0].shape
    truth = np.concatenate(truth)
    predition = np.concatenate(predition)
    truth = to_categorical(truth, nb_classes=6)
    score = label_ranking_average_precision_score(truth, predition)
    print score
    logging.basicConfig(level=logging.INFO)
    logging.info('trueth shape:{}, prediction shape:{}, score:{}'.format(truth.shape, predition.shape, score))
    # results = {'y': truth, 'y_pred': predition}
    # with open('feature_l2_score.pkl', 'w') as f:
    #     cPickle.dump(results, f, protocol=1)

def load_model(model_type):

    if model_type == 'feature_level1':
        _, _, m = model_1.model_face_audio(6)
        # m.load_weights('./model/friend/face_alone_weights.148-0.1980.h5')
        m.load_weights('weights/feature_level_1.hdf5')
    elif model_type == 'feature_level2':
        m = model_1.model_v3(6)
        # ff.load_weights('./model/friend/ff_weights.57-0.0923.h5')
        logging.info('Model {} loading weights...'.format(model_type))
        m.load_weights('weights/weights_feature_level_2.hdf5')
    elif model_type == 'decision_level':
        m = model_1.model_v2(6)
        # m.load_weights('./model/friend/df_weights.126-0.1004.h5') # df2
        # m.load_weights('./model/friend/df_weights.38-0.1629.h5') # df1
        # m.load_weights('weights/new_decision_level_1.hdf5')
        m.load_weights('weights/decision_level_2.hdf5')
    return m

if __name__=='__main__':
    # predict_image()

    m = load_model('decision_level')
    prediction_level(m)