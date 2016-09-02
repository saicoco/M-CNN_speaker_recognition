# -*- coding: utf-8 -*-
# author = sai

import cPickle
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
# import  seaborn as sns
import numpy as np
from scipy import interp
from matplotlib.patches import ConnectionPatch
from keras.utils.np_utils import to_categorical
def roc_auc(test_y, y_score):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    print test_y.shape
    for i in range(6):
        fpr[i], tpr[i], _ = roc_curve(test_y[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(6)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(6):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= 6
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    return fpr['macro'], tpr['macro'], roc_auc['macro']

target_names = ['chandler', 'joey', 'monica', 'phoebe', 'rachel', 'ross']
model_names = ['face_model']

def load_pkl(filename):
    with open(filename, 'r') as f:
        score = cPickle.load(f)
        y, y_pred = score['y'], score['y_pred']
        return y, y_pred

facemodel_y, facemodel_y_pred = load_pkl('model/friend/facemodel_score.pkl')
feature_level2_y, feature_level2_y_pred = load_pkl('model/friend/feature_l2_score.pkl')
feature_level1_y, feature_level1_y_pred = load_pkl('model/friend/feature_l1_score.pkl')
decision_level2_y, decision_level2_y_pred = load_pkl('model/friend/decision_l2_score.pkl')
decision_level1_y, decision_level1_y_pred = load_pkl('model/friend/decision_l1_score.pkl')
src_y, src_y_pred = load_pkl('model/friend/av_src_result.pkl')
imgsrc_y, imgsrc_y_pred = load_pkl('model/friend/img_src_result.pkl')

face_fpr, face_tpr, face_auc = roc_auc(facemodel_y, facemodel_y_pred)
df2_fpr, df2_tpr, df2_auc = roc_auc(decision_level2_y, decision_level2_y_pred)
df1_fpr, df1_tpr, df1_auc = roc_auc(decision_level1_y, decision_level1_y_pred)
f2_fpr, f2_tpr, f2_auc = roc_auc(feature_level2_y, feature_level2_y_pred)
f1_fpr, f1_tpr, f1_auc = roc_auc(feature_level1_y, feature_level1_y_pred)
src_fpr, src_tpr, src_auc = roc_auc(src_y, src_y_pred)
imgsrc_fpr, imgsrc_tpr, imgsrc_auc = roc_auc(imgsrc_y, imgsrc_y_pred)


plt.plot(face_fpr, face_tpr, label='Face Model:(area={:0.4f})'.format(face_auc), linewidth=2)
plt.plot(f1_fpr, f1_tpr, label='Feature Level 1:(area={:0.4f})'.format(f1_auc), linewidth=2)
plt.plot(f2_fpr, f2_tpr, label='Feature Level 2:(area={:0.4f})'.format(f2_auc), linewidth=2)
plt.plot(df1_fpr, df1_tpr, label='Decision Level 1:(area={:0.4f})'.format(df1_auc), linewidth=2)
plt.plot(df2_fpr, df2_tpr, label='Decision Level 2:(area={:0.4f})'.format(df2_auc), linewidth=2)
plt.plot(src_fpr, src_tpr, label='Joint SRC:(area={:0.4f})'.format(src_auc), linewidth=2)
plt.plot(imgsrc_fpr, imgsrc_tpr, label='SRC:(area={:0.4f})'.format(imgsrc_auc), linewidth=2)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.94, 1.0])
plt.axis([ 0.0, 1.0,0.94, 1.0])
plt.grid(True)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve of different Models')
plt.legend(loc="lower right", fontsize=11)
plt.savefig(r'roc.pdf')
plt.show()