# -*- coding: utf-8 -*-
# author = sai
from sklearn.metrics import precision_recall_curve, average_precision_score
import cPickle, numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns

def load_pkl(filename):
    with open(filename, 'r') as f:
        score = cPickle.load(f)
        y, y_pred = score['y'], score['y_pred']
        return y, y_pred

def pred_recall(y, y_pred, title, idx, n_classes=6):
    precision = dict()
    recall = dict()
    average_precision = dict()
    print y.shape
    for i in range(n_classes):
        precision[i], recall[i], thres = precision_recall_curve(y[:, i],
                                                            y_pred[:, i])
        average_precision[i] = average_precision_score(y[:, i], y_pred[:, i])
    target_names = ['chandler', 'joey', 'monica', 'phoebe', 'rachel', 'ross']
    for i in range(n_classes):
        plt.plot(recall[i], precision[i],
                 label='{} (area = {:0.2f})'
                       ''.format(target_names[i], average_precision[i]), linewidth=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    if idx > 4:
        plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.title('{}'.format(title))
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()

facemodel_y, facemodel_y_pred = load_pkl('model/friend/facemodel_score.pkl')
feature_level2_y, feature_level2_y_pred = load_pkl('model/friend/feature_l2_score.pkl')
feature_level1_y, feature_level1_y_pred = load_pkl('model/friend/feature_l1_score.pkl')
decision_level2_y, decision_level2_y_pred = load_pkl('model/friend/decision_l2_score.pkl')
decision_level1_y, decision_level1_y_pred = load_pkl('model/friend/decision_l1_score.pkl')
src_y, src_y_pred = load_pkl('model/friend/av_src_result.pkl')
imgsrc_y, imgsrc_y_pred = load_pkl('model/friend/img_src_result.pkl')

plt.figure(figsize=(16, 9))
p1 = plt.subplot(2, 4, 1)
pred_recall(facemodel_y, facemodel_y_pred, idx=1, title='Face Model')
p2 = plt.subplot(2, 4, 2)
pred_recall(imgsrc_y, imgsrc_y_pred, idx=3, title='SRC')
p3 = plt.subplot(2, 4, 3)
pred_recall(src_y, src_y_pred, idx=2, title='Joint SRC')
p4 = plt.subplot(2, 4, 5)
pred_recall(feature_level1_y, feature_level1_y_pred, idx=5, title='Feature-Level-1 Fusion')
p5 = plt.subplot(2, 4, 6)
pred_recall(feature_level2_y, feature_level2_y_pred, idx=6, title='Feature-Level-2 Fusion')
p6 = plt.subplot(2, 4, 7)
pred_recall(decision_level1_y, decision_level1_y_pred, idx=7, title='Decision-Level-1 Fusion')
p7 = plt.subplot(2, 4, 8)
pred_recall(decision_level2_y, decision_level2_y_pred, idx=8, title='Decision-Level-2 Fusion')

plt.subplots_adjust(wspace=0.3, hspace=0.2)
plt.savefig('pred_recall.pdf')

plt.show()
