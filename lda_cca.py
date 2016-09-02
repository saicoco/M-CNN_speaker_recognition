# -*- coding: utf-8 -*-
# author = sai
import data_flow
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cross_decomposition import CCA
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import SGDClassifier

image_pca = LDA()
audio_pca = LDA()
av_cca = CCA(n_components=3)

def lda_cca(images, audio, label, test=False, fusion_style='concate'):
    # LDA
    if test is False:
        image_pca.fit(images, label)
        audio_pca.fit(audio, label)

    new_img = image_pca.transform(images)
    new_audio = audio_pca.transform(audio)
    # CCA
    if test is False:
        av_cca.fit(new_img, new_audio)

    cca_img = av_cca.transform(new_img)
    cca_audio = av_cca.transform(new_audio)
    if fusion_style is 'concate':
        cca_feature = np.concatenate((cca_img, cca_audio), axis=1)
    else:
        cca_feature = cca_img+cca_audio
    return new_img, new_audio, cca_feature

batch_size = 128
epoches = 50
# bigbang
# train_num = 184100
# test_num = 38584

# friend
train_num = 89833
test_num = 26979.
n_batches = int(train_num/128)
svc = SGDClassifier(alpha=0.0001, l1_ratio=0.15, n_jobs=4, eta0=0.0, power_t=0.5, warm_start=True)
for i in range(epoches):
    av_train = data_flow.av_generator(batch_size=batch_size)
    av_test = data_flow.av_generator(batch_size=test_num, train=False)
    for index in range(n_batches):
        data, label = av_train.next()
        images, audio = data
        images = images.reshape((-1, 3*50*50))
        new_img, new_audio, cca_feature = lda_cca(images, audio, label)

        # param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
        #               'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],}
        # svc = GridSearchCV(SVC(kernel='rbf', class_weight='balanced', max_iter=500), param_grid)
        svc.partial_fit(cca_feature, label, classes=[0, 1, 2, 3, 4, 5])
        # print svc.best_estimator_

    # test
    data, label = av_test.next()
    images, audio = data
    images = images.reshape((-1, 3*50*50))
    test_img, test_audio, test_feature = lda_cca(images, audio, label, test=True)
    pred = svc.predict(test_feature)
    idx = np.where(pred == label)
    print('epoch:{} acc:{}'.format(i, len(idx[0])/test_num))
    print(confusion_matrix(label, pred, labels=range(6)))




