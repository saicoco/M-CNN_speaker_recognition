# -*- coding: utf-8 -*-
# author = sai
import data_flow
import logging
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import normalize
from sklearn.decomposition import MiniBatchDictionaryLearning, SparseCoder
from sklearn.metrics import label_ranking_average_precision_score
import cPickle, sys
from keras.utils.np_utils import to_categorical
logging.basicConfig(level=logging.INFO)

def mini_batch_dl():
    # bigbang
    # train_num = 184100
    # test_num = 38584

    # # friend
    train_num = 89833
    test_num = 26979
    # train_num = 1000
    # test_num = 500
    batch_size = 250
    epoch = 20
    svc = SGDClassifier(loss='log', alpha=0.0001, l1_ratio=0.15, n_iter=1, n_jobs=4, eta0=0.0, power_t=0.5, warm_start=True)
    mini_batch_dl = MiniBatchDictionaryLearning(
        n_components=200,
        alpha=0.15,
        n_iter=1,
        fit_algorithm='lars',
        n_jobs=4,
        batch_size=batch_size,
        shuffle=True
    )
    min_res = 100000000
    acc = []
    for i in xrange(epoch):
        # av_train = data_flow.av_generator(batch_size=batch_size)
        # av_test = data_flow.av_generator(batch_size=batch_size, train=False)
        av_train = data_flow.image_generator(batch_size=batch_size)
        av_test = data_flow.image_generator(batch_size=batch_size, train=False)
        for j in xrange(train_num//batch_size):
            data, labels = av_train.next()
            # images, audio = data
            images = data
            images = images.reshape((images.shape[0], np.prod(images.shape[1:])))
            # I = np.concatenate((images, audio), axis=1)
            I = images
            sys.stdout.write('train...batch/n_batches:{}/{}\r'.format(j*batch_size, train_num))
            sys.stdout.flush()
            mini_batch_dl.partial_fit(X=I)
            images_new = mini_batch_dl.transform(I)
            svc.partial_fit(images_new, labels, classes=[0, 1, 2, 3, 4, 5])
        # coder
        dictionary = normalize(mini_batch_dl.components_, norm='l2')
        sparse_coder = SparseCoder(dictionary=dictionary, n_jobs=4)
        for k in xrange(test_num // batch_size):
            test_old, labels = av_test.next()
            # images, audio = test_old
            images = test_old
            images = images.reshape((images.shape[0], np.prod(images.shape[1:])))
            # I = np.concatenate((images, audio), axis=1)
            I = images
            images_new = sparse_coder.transform(I)
            acc.append(svc.score(images_new, labels))
            images_new = np.dot(images_new, dictionary)
            res = np.linalg.norm((images_new-I))
            sys.stdout.write('test...batch/n_batches:{}/{}, acc:{}, res:{}\r'.format(k * batch_size, test_num, acc[-1], res))
            sys.stdout.flush()
            if res < min_res:
                best_dict = dictionary
                best_svc = svc
        logging.info('epoch:{}, res_error:{}, accuracy:{}'.format(i, res, np.mean(acc)))
        with open('imgsrc_dict_logistic_{}.pkl'.format(i), 'w') as f:
            cPickle.dump(best_dict, f, protocol=1)
        with open('imgbes_svc_logistic_{}.pkl'.format(i), 'w') as f:
            cPickle.dump(best_svc, f, protocol=1)
        logging.info('save dictionary & classifier')

def predict():
    test_num = 26979
    batch_size = 250
    av_test = data_flow.av_generator(batch_size=batch_size, train=False)
    # av_test = data_flow.image_generator(batch_size=batch_size, train=False)
    with open('JSRC/jsrc_dict_logistic_5.pkl', 'r') as f:
        dictionary = cPickle.load(f)
        print dictionary.shape
    with open('JSRC/bes_svc_logistic_5.pkl', 'r') as f:
        svc = cPickle.load(f)
        print svc
    sparse_coder = SparseCoder(dictionary=dictionary, n_jobs=4)
    acc = []
    predict_res = []
    truth = []
    for k in xrange(test_num // batch_size):
        test_old, labels = av_test.next()
        images, audio = test_old
        # images = test_old
        images = images.reshape((images.shape[0], np.prod(images.shape[1:])))
        I = np.concatenate((images, audio), axis=1)
        # I = images
        images_new = sparse_coder.transform(I)
        predict_res.append(svc.predict_proba(images_new))
        truth.append(labels)
        # acc.append(svc.score(images_new, labels))
        # images_new = np.dot(images_new, dictionary)
        # res = np.linalg.norm((images_new - I))
        # sys.stdout.write(
        #     'test...batch/n_batches:{}/{}, acc:{}, res:{}\r'.format(k * batch_size, test_num, acc[-1], res))
        # sys.stdout.flush()

    predict_res = np.concatenate(predict_res)
    truth = np.concatenate(truth)
    truth = to_categorical(truth, nb_classes=6)
    print predict_res[0].shape, truth.shape
    score = label_ranking_average_precision_score(truth, predict_res)
    print 'score:', score
    # result = {'y':truth, 'y_pred':predict_res}
    # with open('model/friend/img_src_result.pkl', 'w') as f:
    #     cPickle.dump(result, f, protocol=1)
if __name__=='__main__':
    # mini_batch_dl()
    predict()